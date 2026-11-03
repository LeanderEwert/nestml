# -*- coding: utf-8 -*-
#
# test__gpu_compartmental_exhaustive_benchmark.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

"""Exhaustive GPU compartmental benchmark: model variant x morphology x dynamics x size.

Standalone driver (not part of the pytest suite). Sweeps every model configuration
(solver x fastexp) against every morphology (chain, star, capacity-based binary tree)
under both active and passive dynamics across a shared size axis, comparing generated
NEST-GPU runtime and trace accuracy against a single-precision CPU NESTML reference.

Results are written to one JSON entry per (config, morphology, dynamics, size); the
single-precision CPU reference is config-independent and shared across all four GPU
configs. See _exhaustive_json_path and _cpu_reference_json_path.

Run directly, e.g.::

    EXHAUSTIVE_SIZES=1,2,4 python test__gpu_compartmental_exhaustive_benchmark.py

A Rich progress view shows two bars pinned to the bottom of the terminal: one over
the 40 setups (config x morphology x dynamics) and one over the shared size axis,
with intermediate timing results scrolling above them.
"""

import argparse
import contextlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

import nest
import numpy as np
from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target

TESTS_PATH = os.path.realpath(os.path.dirname(__file__))
if TESTS_PATH not in sys.path:
    sys.path.insert(0, TESTS_PATH)

from gpu_compartmental_model_runner import (
    benchmark_neuron_configuration,
    generate_binary_tree_parents,
)
from test__gpu_compartmental_model import (
    DT,
    PLOT_IMPORT_ERROR,
    SIM_TIME,
    TEST_PLOTS,
    compare_trace,
    cpu_stimulus_spike_times,
    generate_gpu_default_model,
)

CPU_SINGLE_PRECISION_MODEL = "cm_default_cpu_single_nestml"
CPU_SINGLE_PRECISION_MODULE = "cm_default_cpu_single_module"
# Module-scoped cache so the CPU reference is built once per fastexp value even
# though benchmark_config is parametrized over four configs (two per value).
_CPU_REFERENCE_CACHE = {}

# Cache for single-precision CPU reference runs keyed by
# (filename_key, dynamics, n_added_compartments). The reference is config-independent
# (single precision + fastexp) so it is computed once per morphology variation and
# reused across all four GPU configs.
_CPU_RUN_CACHE = {}

# Stable location for per-entry benchmark JSON. The GPU module is rebuilt in place
# (shutil.rmtree) at target_path for every config, so results must not live there.
RESULTS_PATH = os.path.join(TESTS_PATH, "exhaustive_benchmark_results")


# Model configurations: full cross-product of solver {hines, r_edd} x fastexp
# {default, fastexp}. use_fastexp only changes generated propagator code, not the
# installed model name, so default/fastexp GPU variants share one install.
MODEL_CONFIGS = [
    {"name": "default+hines", "solver": "hines", "use_fastexp": False},
    {"name": "default+r_edd", "solver": "r_edd", "use_fastexp": False},
    {"name": "fastexp+hines", "solver": "hines", "use_fastexp": True},
    {"name": "fastexp+r_edd", "solver": "r_edd", "use_fastexp": True},
]

# Morphologies. The tree variants use a capacity-based binary tree generator with a
# distinct branching probability (per-node capacity of 1 or 2) and a fixed seed so
# each variant is a deterministic, comparable morphology.
MORPHOLOGIES = [
    {"name": "chain"},
    {"name": "star"},
    {"name": "tree", "branching_prob": 1.0, "seed": 0},
    {"name": "tree", "branching_prob": 0.5, "seed": 1},
    {"name": "tree", "branching_prob": 0.1, "seed": 2},
]

DYNAMICS = ["active", "passive"]

# Shared complexity axis: neuron count for chain/star, added dendritic compartments
# for trees. Same power-of-two axis so morphologies are comparable across complexity.
# Override with EXHAUSTIVE_SIZES (comma-separated, e.g. "1,2,4,8") for a quick smoke run.
def _load_sizes():
    raw = os.environ.get("EXHAUSTIVE_SIZES")
    if not raw:
        return [2 ** i for i in range(16)]
    return [int(x) for x in raw.split(",")]


def _load_configs():
    """Filter MODEL_CONFIGS by EXHAUSTIVE_CONFIGS (comma-separated names) for scoped runs.

    Lets a smoke run target a single solver/fastexp variant without building all four
    GPU modules, which is important under tight memory pressure where building several
    large chain modules back-to-back can OOM.
    """
    raw = os.environ.get("EXHAUSTIVE_CONFIGS")
    if not raw:
        return MODEL_CONFIGS
    wanted = {name.strip() for name in raw.split(",")}
    filtered = [c for c in MODEL_CONFIGS if c["name"] in wanted]
    if not filtered:
        raise SystemExit(
            f"EXHAUSTIVE_CONFIGS={raw!r} matched no known configs: "
            f"{[c['name'] for c in MODEL_CONFIGS]}")
    return filtered


SIZES = _load_sizes()
MODEL_CONFIGS = _load_configs()

BENCHMARK_RANDOM_SEED = 12345
COMPARTMENT_BENCHMARK_SPIKE_TIMES = [10.0, 13.0, 16.0]
SKIP_REBUILD_ENV = "NESTML_GPU_CM_SKIP_REBUILD"

# Number of cells drawn per progress bar (setup bar spans all four configs).
_SETUP_TOTAL = len(MODEL_CONFIGS) * len(MORPHOLOGIES) * len(DYNAMICS)
_SETUP_ORDER = [(c, m, d) for c in MODEL_CONFIGS for m in MORPHOLOGIES for d in DYNAMICS]


if TEST_PLOTS:
    import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Live progress view
# ---------------------------------------------------------------------------
# Module-level Rich console + progress view. Two tasks give two bars pinned to the
# bottom of the terminal: OUTER_TASK sweeps the 40 setups (config x morphology x
# dynamics), INNER_TASK sweeps the shared size axis. Intermediate messages are printed
# to the same console so they scroll above the bars.
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

console = Console()
progress = Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    console=console,
    transient=False,
    expand=True,
)
OUTER_TASK = None
INNER_TASK = None


def _set_outer(done, label):
    if OUTER_TASK is not None:
        progress.update(OUTER_TASK, completed=done, description=label)


def _set_inner(done, label):
    if INNER_TASK is not None:
        progress.update(INNER_TASK, completed=done, description=label)


# ---------------------------------------------------------------------------
# Model building and simulation
# ---------------------------------------------------------------------------
def _build_cpu_reference():
    """Single-precision cm_default CPU module, built once with fastexp.

    The reference is config-independent and only serves as a stable GPU/CPU runtime
    reference, so its fastexp setting must not track the GPU config's fastexp flag.
    Building here (rather than requesting the parametrized cpu fixture) avoids pytest's
    fixture-parametrization product, which would otherwise rebuild the GPU module twice
    per config.
    """
    if True in _CPU_REFERENCE_CACHE:
        return _CPU_REFERENCE_CACHE[True]

    tests_path = os.path.realpath(os.path.dirname(__file__))
    input_path = os.path.join(
        tests_path, "..", "nest_compartmental_tests", "resources", "cm_default.nestml")
    target_path = os.path.join(tests_path, "target_cpu_single_precision_fastexp")
    install_path = os.path.join(target_path, "install")
    module_path = os.path.join(install_path, f"{CPU_SINGLE_PRECISION_MODULE}.so")

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        assert os.path.isfile(module_path), (
            f"{SKIP_REBUILD_ENV} is set, but the CPU reference module does not exist at {module_path}")
        console.print(f"Skipping single-precision CPU cm_default rebuild because {SKIP_REBUILD_ENV} is set",
              style="blue")
    else:
        os.makedirs(install_path, exist_ok=True)
        generate_nest_compartmental_target(
            input_path=input_path,
            target_path=target_path,
            install_path=install_path,
            module_name=CPU_SINGLE_PRECISION_MODULE,
            suffix="_cpu_single_nestml",
            logging_level="ERROR",
            codegen_opts={
                "fp_precision": "single",
                "nest_version": NESTTools.detect_nest_version(),
                "use_fastexp": True,
            },
        )

    reference = {"variant": "fastexp", "model_name": CPU_SINGLE_PRECISION_MODEL, "module_path": module_path}
    _CPU_REFERENCE_CACHE[True] = reference
    return reference


def run_cpu_active_compartment_cm_default(
        cpu_model, n_added_compartments, sample_compartment, record=True, morphology="chain",
        dynamics=None, parents=None):
    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    recordables = [f"v_comp{sample_compartment}"]

    t_start = time.perf_counter()
    nest.ResetKernel()
    nest.Install(cpu_model["module_path"])
    nest.SetKernelStatus({"resolution": DT})

    neuron = nest.Create(cpu_model["model_name"])
    neuron.set(benchmark_neuron_configuration(
        n_added_compartments, morphology, dynamics, parents=parents))

    spike_generator = nest.Create("spike_generator", 1, {
        "spike_times": cpu_stimulus_spike_times(COMPARTMENT_BENCHMARK_SPIKE_TIMES)})

    nest.Connect(spike_generator, neuron, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})

    if record:
        multimeter = nest.Create("multimeter", 1, {"record_from": recordables, "interval": DT})
        nest.Connect(multimeter, neuron)

    nest.Simulate(SIM_TIME)
    traces = nest.GetStatus(multimeter, "events")[0] if record else None
    runtime = time.perf_counter() - t_start

    result = {
        "n_added_compartments": n_added_compartments,
        "n_compartments": n_added_compartments + 1,
        "sample_compartment": sample_compartment,
        "morphology": morphology,
        "recording_enabled": record,
        "dynamics": dynamics,
        "runtime": runtime,
    }
    if record:
        result["traces"] = traces
    return result


def run_gpu_active_compartment_cm_default(
        tmp_path, n_added_compartments, sample_compartment, record=True, morphology="chain",
        dynamics=None, parents=None):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    prefix = f"{dynamics}-" + ("tree-" if morphology == "tree" else "")
    mode = prefix + ("compartment-json" if record else "compartment-no-record-json")
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / (
        f"cm_default_gpu_{dynamics}_{morphology}_compartment_{n_added_compartments}_{recording_suffix}.json")
    args = [
        sys.executable,
        runner,
        mode,
        str(output_path),
        str(n_added_compartments),
        str(sample_compartment),
    ]
    if parents is not None:
        args.append(json.dumps(parents))
    subprocess.run(args, check=True)
    with open(output_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def compare_compartment_traces(cpu, gpu, compartment, dynamics):
    variable = f"v_comp{compartment}"
    atol = 2.0 if dynamics == "active" and compartment == 0 else 0.5
    compare_trace(cpu["traces"], gpu["traces"], variable, variable, atol=atol)


def tree_parents_for(morphology, n):
    if morphology["name"] != "tree":
        return None
    return generate_binary_tree_parents(n, morphology["branching_prob"], morphology["seed"])


def _morphology_filename_key(morphology_def):
    """Unique, filesystem-safe key identifying a morphology variant.

    Chain and star have a single variant; the three tree variants are
    distinguished by branching probability so each produces its own results file.
    """
    name = morphology_def["name"]
    branching_prob = morphology_def.get("branching_prob")
    if branching_prob is None:
        return name
    return f"{name}_bp{branching_prob}"


def _exhaustive_json_path(config, morphology, dynamics):
    filename_key = _morphology_filename_key(morphology)
    return os.path.join(
        RESULTS_PATH,
        f"exhaustive_{config['name']}_{filename_key}_{dynamics}_benchmark.json")


def _load_entry(config, morphology, dynamics, n):
    json_path = _exhaustive_json_path(config, morphology, dynamics)
    if not os.path.isfile(json_path):
        return None
    seed = morphology.get("seed")
    branching_prob = morphology.get("branching_prob")
    with open(json_path, encoding="utf-8") as input_file:
        entries = json.load(input_file)
    for entry in entries:
        if entry["n_added_compartments"] == n and entry["seed"] == seed and entry["branching_prob"] == branching_prob:
            return entry
    return None


def _save_entry(config, morphology, dynamics, entry):
    json_path = _exhaustive_json_path(config, morphology, dynamics)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    entries = []
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as input_file:
            entries = json.load(input_file)
    for i, existing in enumerate(entries):
        if (existing["n_added_compartments"] == entry["n_added_compartments"]
                and existing["seed"] == entry["seed"]
                and existing["branching_prob"] == entry["branching_prob"]):
            entries[i] = entry
            break
    else:
        entries.append(entry)
    entries.sort(key=lambda e: e["n_added_compartments"])
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)


# The single-precision CPU reference is config-independent (single precision + fastexp),
# so it is computed once per morphology variation and shared across all four GPU configs.
# It lives in its own config-independent JSON file so it survives the per-config rmtree
# of target_path and is reused across separate runs.
def _cpu_reference_json_path(morphology, dynamics):
    filename_key = _morphology_filename_key(morphology)
    return os.path.join(RESULTS_PATH, f"exhaustive_cpu_{filename_key}_{dynamics}_benchmark.json")


def _serializable_traces(traces):
    # NEST multimeter events hold numpy arrays; the GPU runner stores plain lists, so
    # convert on write to keep the shared reference JSON-serializable and comparable.
    return {key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in traces.items()}


def _load_cpu_reference(morphology, dynamics, n):
    cache_key = (_morphology_filename_key(morphology), dynamics, n)
    if cache_key in _CPU_RUN_CACHE:
        return _CPU_RUN_CACHE[cache_key]
    json_path = _cpu_reference_json_path(morphology, dynamics)
    if not os.path.isfile(json_path):
        return None
    seed = morphology.get("seed")
    branching_prob = morphology.get("branching_prob")
    with open(json_path, encoding="utf-8") as input_file:
        entries = json.load(input_file)
    for entry in entries:
        if entry["n_added_compartments"] == n and entry["seed"] == seed and entry["branching_prob"] == branching_prob:
            _CPU_RUN_CACHE[cache_key] = entry
            return entry
    return None


def _save_cpu_reference(morphology, dynamics, entry):
    entry = dict(entry)
    if "traces" in entry:
        entry["traces"] = _serializable_traces(entry["traces"])
    json_path = _cpu_reference_json_path(morphology, dynamics)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    seed = morphology.get("seed")
    branching_prob = morphology.get("branching_prob")
    entries = []
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as input_file:
            entries = json.load(input_file)
    for i, existing in enumerate(entries):
        if (existing["n_added_compartments"] == entry["n_added_compartments"]
                and existing["seed"] == seed and existing["branching_prob"] == branching_prob):
            entries[i] = entry
            break
    else:
        entries.append(entry)
    entries.sort(key=lambda e: e["n_added_compartments"])
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)
    _CPU_RUN_CACHE[_morphology_filename_key(morphology), dynamics, entry["n_added_compartments"]] = entry


def _print_entry(title, entry):
    cpu_runtime = entry.get("cpu_runtime")
    if cpu_runtime is None:
        cpu_ref = _load_cpu_reference(
            {"name": entry["morphology"], "branching_prob": entry.get("branching_prob"), "seed": entry.get("seed")},
            entry["dynamics"], entry["n_added_compartments"])
        cpu_runtime = cpu_ref["cpu_runtime"] if cpu_ref else float("nan")
    console.print(f"{title}: {entry['morphology']} {entry['dynamics']} "
          f"n={entry['n_added_compartments']:<6} "
          f"sample={entry['sample_compartment']:<6} "
          f"cpu={cpu_runtime:.6f}s gpu={entry['gpu_runtime']:.6f}s "
          f"ratio={entry['gpu_runtime'] / cpu_runtime:.3f}")


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def _build_gpu_model(config):
    variant = "fastexp" if config["use_fastexp"] else "default"
    target_path = os.path.join(TESTS_PATH, "target")
    nest_gpu_path = os.environ.get("NEST_GPU", os.getcwd())
    assert os.path.isdir(os.path.join(nest_gpu_path, "src")), (
        f"NEST-GPU source tree not found near {nest_gpu_path!r}; set NEST_GPU to it")

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        os.makedirs(target_path, exist_ok=True)
        console.print(f"Skipping cm_default_nestml rebuild because {SKIP_REBUILD_ENV} is set", style="blue")
    else:
        generate_gpu_default_model(target_path, codegen_opts={
            "gpu_tree_solver": config["solver"],
            "gpu_tree_solver_chain_length": 3,
            "gpu_tree_solver_base_size": 32,
            "use_fastexp": config["use_fastexp"],
        })
    return {"config": config, "variant": variant, "cpu": _build_cpu_reference(), "gpu_path": target_path}


def _run_one(config, morphology_def, dynamics, size, tmp_path):
    variant = "fastexp" if config["use_fastexp"] else "default"
    morphology = morphology_def["name"]
    entry = _load_entry(config, morphology_def, dynamics, size)
    if entry is not None:
        _print_entry(f"reusing cached {variant} result", entry)
        return entry

    parents = tree_parents_for(morphology_def, size)
    sample_compartment = int(np.random.default_rng(BENCHMARK_RANDOM_SEED).integers(0, size + 1))

    cpu = _load_cpu_reference(morphology_def, dynamics, size)
    if cpu is None:
        cpu_model = _build_cpu_reference()
        console.print(f"running single-precision CPU NESTML reference simulation (fastexp) [{morphology}/{dynamics} n={size}]")
        cpu = run_cpu_active_compartment_cm_default(
            cpu_model, size, sample_compartment,
            dynamics=dynamics, parents=parents)
        cpu["cpu_runtime"] = cpu["runtime"]
        console.print("running single-precision CPU NESTML simulation without recording")
        cpu_no_record = run_cpu_active_compartment_cm_default(
            cpu_model, size, sample_compartment, record=False,
            dynamics=dynamics, parents=parents)
        cpu["cpu_no_record_runtime"] = cpu_no_record["runtime"]
        cpu["seed"] = morphology_def.get("seed")
        cpu["branching_prob"] = morphology_def.get("branching_prob")
        _save_cpu_reference(morphology_def, dynamics, cpu)

    console.print(f"running generated NEST-GPU simulation [{config['name']} {morphology} {dynamics} n={size}]")
    gpu = run_gpu_active_compartment_cm_default(
        tmp_path, size, sample_compartment,
        morphology=morphology, dynamics=dynamics, parents=parents)
    compare_compartment_traces(cpu, gpu, sample_compartment, dynamics)
    console.print("running generated NEST-GPU simulation without recording")
    gpu_no_record = run_gpu_active_compartment_cm_default(
        tmp_path, size, sample_compartment, record=False,
        morphology=morphology, dynamics=dynamics, parents=parents)

    entry = {
        "config": config["name"],
        "variant": variant,
        "morphology": morphology,
        "branching_prob": morphology_def.get("branching_prob"),
        "seed": morphology_def.get("seed"),
        "n_added_compartments": size,
        "n_compartments": size + 1,
        "sample_compartment": sample_compartment,
        "dynamics": dynamics,
        "gpu_runtime": gpu["runtime"],
        "gpu_no_record_runtime": gpu_no_record["runtime"],
    }
    _save_entry(config, morphology_def, dynamics, entry)
    _print_entry(f"finished {variant} result", entry)
    return entry


def _load_exhaustive_results(config_name, morphology_def, dynamics):
    filename_key = _morphology_filename_key(morphology_def)
    json_path = os.path.join(
        RESULTS_PATH, f"exhaustive_{config_name}_{filename_key}_{dynamics}_benchmark.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as input_file:
        return json.load(input_file)


# Distinct line colors for the four model configurations, reused across every cell.
CONFIG_COLORS = {
    "default+hines": "#1f77b4",
    "default+r_edd": "#ff7f0e",
    "fastexp+hines": "#2ca02c",
    "fastexp+r_edd": "#d62728",
}


def _legend_handles():
    import matplotlib.lines as mlines
    return [
        mlines.Line2D([], [], marker="o", lw=1.5, color=CONFIG_COLORS[c["name"]], label=c["name"])
        for c in MODEL_CONFIGS
    ]


def _morphology_label(morphology_def):
    name = morphology_def["name"]
    branching_prob = morphology_def.get("branching_prob")
    if branching_prob is None:
        return name
    return f"{name} (p={branching_prob})"


def _cpu_runtimes_by_size(morphology_def, dynamics):
    json_path = _cpu_reference_json_path(morphology_def, dynamics)
    if not os.path.isfile(json_path):
        return {}
    seed = morphology_def.get("seed")
    branching_prob = morphology_def.get("branching_prob")
    with open(json_path, encoding="utf-8") as input_file:
        entries = json.load(input_file)
    return {
        entry["n_added_compartments"]: entry["cpu_runtime"]
        for entry in entries
        if entry["seed"] == seed and entry["branching_prob"] == branching_prob
    }


def _plot_cell(ax, morphology_def, dynamics):
    cpu_runtimes_by_size = _cpu_runtimes_by_size(morphology_def, dynamics)
    for config in MODEL_CONFIGS:
        results = _load_exhaustive_results(config["name"], morphology_def, dynamics)
        if not results:
            continue
        x = []
        relative_runtimes = []
        for r in results:
            cpu_runtime = cpu_runtimes_by_size.get(r["n_added_compartments"])
            if cpu_runtime is None or cpu_runtime == 0:
                continue
            x.append(r["n_added_compartments"])
            relative_runtimes.append(r["gpu_runtime"] / cpu_runtime)
        if not x:
            continue
        x = np.asarray(x, dtype=int)
        relative_runtimes = np.asarray(relative_runtimes, dtype=float)
        ax.plot(x, relative_runtimes, marker="o", lw=1.5,
                color=CONFIG_COLORS[config["name"]], label=config["name"])
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("added dendritic compartment count")
    ax.set_ylabel("GPU / CPU runtime")
    ax.set_title(f"{_morphology_label(morphology_def)} · {dynamics}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize="x-small")


def plot_all_exhaustive(target_path):
    if not TEST_PLOTS:
        console.print(f"Skipping GPU compartmental exhaustive benchmark plots: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(target_path, exist_ok=True)
    output_path = os.path.join(target_path, "exhaustive_benchmark_all_configs.png")
    console.print(f"Writing GPU compartmental exhaustive benchmark plots to {output_path}")

    n_rows = len(MORPHOLOGIES)
    n_cols = len(DYNAMICS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 3 * n_rows), squeeze=False)
    for row, morphology_def in enumerate(MORPHOLOGIES):
        for col, dynamics in enumerate(DYNAMICS):
            _plot_cell(axes[row][col], morphology_def, dynamics)
    fig.suptitle("cm_default exhaustive benchmark: GPU/CPU relative runtime, all configs", fontsize=14)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, fontsize="small", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _exhaustive_plot():
    target_path = os.path.join(TESTS_PATH, "target")
    plot_all_exhaustive(target_path)


def main(argv=None):
    global OUTER_TASK, INNER_TASK
    parser = argparse.ArgumentParser(description="Exhaustive GPU compartmental benchmark")
    parser.parse_args(argv)

    progress.start()
    OUTER_TASK = progress.add_task("[cyan]setups", total=_SETUP_TOTAL, start=False)
    INNER_TASK = progress.add_task("[green]sizes", total=len(SIZES), start=False)

    tmp_path = pathlib.Path(TESTS_PATH) / "exhaustive_benchmark_tmp"
    tmp_path.mkdir(exist_ok=True)

    console.print("Exhaustive GPU compartmental benchmark", style="bold")
    console.print(f"morphologies={len(MORPHOLOGIES)} dynamics={len(DYNAMICS)} "
          f"configs={len(MODEL_CONFIGS)} sizes={len(SIZES)} "
          f"({_SETUP_TOTAL} setups x {len(SIZES)} sizes)", style="dim")

    setup_index = 0
    try:
        for config in MODEL_CONFIGS:
            _build_gpu_model(config)
            for morphology_def in MORPHOLOGIES:
                for dynamics in DYNAMICS:
                    setup_index += 1
                    label = f"{config['name']} {morphology_def['name']} {dynamics}"
                    _set_outer(setup_index, label)
                    for size_index, size in enumerate(SIZES, start=1):
                        _run_one(config, morphology_def, dynamics, size, tmp_path)
                        _set_inner(size_index, f"n={size}")
                    _set_outer(setup_index, label)
    except Exception as exc:
        console.print(f"\nbenchmark aborted: {exc}", style="bold red")
        raise
    finally:
        progress.stop()

    console.print("generating plots")
    _exhaustive_plot()
    console.print("done", style="bold green")


if __name__ == "__main__":
    main()
