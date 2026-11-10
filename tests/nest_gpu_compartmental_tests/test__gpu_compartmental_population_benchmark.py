# -*- coding: utf-8 -*-
#
# test__gpu_compartmental_population_benchmark.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

"""Population/chain/star sweeps (1-D) and a neurons x compartments sweep (2-D), each
with active and passive dynamics.

Both variants use somatic AMPA only and identical voltage recording. Run in
~/.venvs/nestml-gpu; pytest -k active or -k passive selects a dynamics variant.
Runtime includes setup and one simulation, excluding trace retrieval.

The tree solver is selected via ``NESTML_GPU_CM_SOLVER`` (default ``r_edd``) and the
propagator variant via ``NESTML_GPU_CM_VARIANT`` (default ``default``). Named entry
functions ``run_benchmark_1d`` / ``run_benchmark_2d`` drive both sweeps and can be run
directly (see the ``__main__`` block) without pytest.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import nest
import numpy as np
import pytest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target

TESTS_PATH = os.path.realpath(os.path.dirname(__file__))
if TESTS_PATH not in sys.path:
    sys.path.insert(0, TESTS_PATH)

from test__gpu_compartmental_model import (  # noqa: E402
    DT,
    PLOT_IMPORT_ERROR,
    SIM_TIME,
    TEST_PLOTS,
    compare_trace,
    cpu_stimulus_spike_times,
    generate_gpu_default_model,
)

from gpu_compartmental_model_runner import benchmark_neuron_configuration  # noqa: E402


BENCHMARK_POPULATION_SIZES = [2 ** (i*1) for i in range(13)]
BENCHMARK_COMPARTMENT_SIZES = [2 ** (i*1) for i in range(13)]
BENCHMARK_RANDOM_SEED = 12345
COMPARTMENT_BENCHMARK_SPIKE_TIMES = [10.0, 13.0, 16.0]
SKIP_REBUILD_ENV = "NESTML_GPU_CM_SKIP_REBUILD"
SOLVER_ENV = "NESTML_GPU_CM_SOLVER"
DEFAULT_SOLVER = "r_edd"
VARIANT_ENV = "NESTML_GPU_CM_VARIANT"
DEFAULT_VARIANT = "default"

CPU_SINGLE_PRECISION_MODEL = "cm_default_cpu_single_nestml"
CPU_SINGLE_PRECISION_MODULE = "cm_default_cpu_single_module"

if TEST_PLOTS:
    import matplotlib.pyplot as plt


def _build_gpu_target(solver, variant):
    """Generate + install the cm_default NEST-GPU module for one solver/variant.

    ``use_fastexp`` only changes the generated propagator code, not the installed model
    name, so the ``default`` and ``fastexp`` variants share a single install. Skipped
    (target dir created empty) when ``NESTML_GPU_CM_SKIP_REBUILD`` is set.
    """
    target_path = os.path.join(TESTS_PATH, "target")
    nest_gpu_path = os.environ.get("NEST_GPU", os.getcwd())
    assert os.path.isdir(os.path.join(nest_gpu_path, "src"))

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        os.makedirs(target_path, exist_ok=True)
        print(f"Skipping cm_default_nestml rebuild because {SKIP_REBUILD_ENV} is set")
    else:
        generate_gpu_default_model(target_path, codegen_opts={
            "gpu_tree_solver": solver,
            "gpu_tree_solver_chain_length": 3,
            "gpu_tree_solver_base_size": 32,
            "use_fastexp": variant == "fastexp",
        })

    return target_path


def _build_cpu_model(variant):
    """Single-precision cm_default CPU module, mirroring the ``cpu_model_variant`` fixture.

    ``use_fastexp`` only changes the generated propagator code, not the installed model
    name, so both variants share a module name but use a variant-specific target path.
    Skipped (module asserted present) when ``NESTML_GPU_CM_SKIP_REBUILD`` is set.
    """
    tests_path = os.path.realpath(os.path.dirname(__file__))
    input_path = os.path.join(
        tests_path, "..", "nest_compartmental_tests", "resources", "cm_default.nestml")
    target_path = os.path.join(tests_path, f"target_cpu_single_precision_{variant}")
    install_path = os.path.join(target_path, "install")
    module_path = os.path.join(install_path, f"{CPU_SINGLE_PRECISION_MODULE}.so")

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        assert os.path.isfile(module_path), (
            f"{SKIP_REBUILD_ENV} is set, but the CPU reference module does not exist at {module_path}")
        print(f"Skipping single-precision CPU cm_default rebuild because {SKIP_REBUILD_ENV} is set")
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
                "use_fastexp": variant == "fastexp",
            },
        )

    return {
        "variant": variant,
        "model_name": CPU_SINGLE_PRECISION_MODEL,
        "module_path": module_path,
    }


@pytest.fixture(scope="module", params=["default", "fastexp"])
def benchmark_models(request):
    """cm_default CPU + NEST-GPU modules for one benchmark variant.

    The GPU module is rebuilt in place for each variant so the ``default`` and
    ``fastexp`` variants share a single install. Parametrizing here (rather than adding
    a second fixture) keeps both variants aligned. The tree solver is taken from
    ``NESTML_GPU_CM_SOLVER`` (default ``r_edd``).
    """
    variant = request.param
    solver = os.environ.get(SOLVER_ENV, DEFAULT_SOLVER)
    target_path = _build_gpu_target(solver, variant)

    return {
        "variant": variant,
        "solver": solver,
        "cpu": request.getfixturevalue("cpu_model_variant"),
        "gpu_path": target_path,
    }


@pytest.fixture(scope="module")
def benchmark_target_path():
    # Retained for the profiling module, which only needs the default variant.
    solver = os.environ.get(SOLVER_ENV, DEFAULT_SOLVER)
    return _build_gpu_target(solver, DEFAULT_VARIANT)


def run_cpu_active_population_cm_default(cpu_model, n_neurons, sample_neuron, record=True, dynamics="active"):
    nest.ResetKernel()
    nest.Install(cpu_model["module_path"])
    nest.SetKernelStatus({"resolution": DT})

    t_start = time.perf_counter()
    neurons = nest.Create(cpu_model["model_name"], n_neurons)
    neurons.set(benchmark_neuron_configuration(1, dynamics=dynamics))

    sg_soma = nest.Create("spike_generator", n_neurons, {
        "spike_times": cpu_stimulus_spike_times(COMPARTMENT_BENCHMARK_SPIKE_TIMES)})

    nest.Connect(sg_soma, neurons, conn_spec={"rule": "one_to_one"}, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})

    if record:
        multimeter = nest.Create(
            "multimeter", 1, {"record_from": ["v_comp0", "v_comp1"], "interval": DT})
        nest.Connect(multimeter, neurons[sample_neuron])

    nest.Simulate(SIM_TIME)
    runtime = time.perf_counter() - t_start

    result = {
        "n_neurons": n_neurons,
        "sample_neuron": sample_neuron,
        "recording_enabled": record,
        "dynamics": dynamics,
        "runtime": runtime,
    }
    if record:
        result["traces"] = nest.GetStatus(multimeter, "events")[0]
    return result


def run_gpu_active_population_cm_default(tmp_path, n_neurons, sample_neuron, record=True, dynamics="active"):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    mode = f"{dynamics}-population-json" if record else f"{dynamics}-population-no-record-json"
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / f"cm_default_gpu_{dynamics}_population_{n_neurons}_{recording_suffix}.json"
    subprocess.check_call([
        sys.executable,
        runner,
        mode,
        str(output_path),
        str(n_neurons),
        str(sample_neuron),
    ])
    with open(output_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def run_cpu_active_compartment_cm_default(
        cpu_model, n_added_compartments, sample_compartment, record=True, morphology="chain", dynamics=None):
    nest.ResetKernel()
    nest.Install(cpu_model["module_path"])
    nest.SetKernelStatus({"resolution": DT})

    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    recordables = [f"v_comp{sample_compartment}"]

    t_start = time.perf_counter()
    neuron = nest.Create(cpu_model["model_name"])
    neuron.set(benchmark_neuron_configuration(n_added_compartments, morphology, dynamics))

    spike_generator = nest.Create("spike_generator", 1, {
        "spike_times": cpu_stimulus_spike_times(COMPARTMENT_BENCHMARK_SPIKE_TIMES)})

    nest.Connect(spike_generator, neuron, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})

    if record:
        multimeter = nest.Create("multimeter", 1, {"record_from": recordables, "interval": DT})
        nest.Connect(multimeter, neuron)

    nest.Simulate(SIM_TIME)
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
        result["traces"] = nest.GetStatus(multimeter, "events")[0]
    return result


def run_cpu_active_compartment_population_cm_default(
        cpu_model, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        record=True, morphology="chain", dynamics=None):
    nest.ResetKernel()
    nest.Install(cpu_model["module_path"])
    nest.SetKernelStatus({"resolution": DT})

    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    if sample_neuron < 0 or sample_neuron >= n_neurons:
        raise ValueError("sample_neuron must be inside the created population")
    recordables = [f"v_comp{sample_compartment}"]

    t_start = time.perf_counter()
    neurons = nest.Create(cpu_model["model_name"], n_neurons)
    neurons.set(benchmark_neuron_configuration(n_added_compartments, morphology, dynamics))

    spike_generator = nest.Create("spike_generator", n_neurons, {
        "spike_times": cpu_stimulus_spike_times(COMPARTMENT_BENCHMARK_SPIKE_TIMES)})

    nest.Connect(spike_generator, neurons, conn_spec={"rule": "one_to_one"}, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})

    if record:
        multimeter = nest.Create("multimeter", 1, {"record_from": recordables, "interval": DT})
        nest.Connect(multimeter, neurons[sample_neuron])

    nest.Simulate(SIM_TIME)
    runtime = time.perf_counter() - t_start

    result = {
        "n_neurons": n_neurons,
        "n_added_compartments": n_added_compartments,
        "n_compartments": n_added_compartments + 1,
        "sample_neuron": sample_neuron,
        "sample_compartment": sample_compartment,
        "morphology": morphology,
        "recording_enabled": record,
        "dynamics": dynamics,
        "runtime": runtime,
    }
    if record:
        result["traces"] = nest.GetStatus(multimeter, "events")[0]
    return result


def run_gpu_active_compartment_population_cm_default(
        tmp_path, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        record=True, morphology="chain", dynamics=None):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    if morphology not in ("chain", "star"):
        raise ValueError(f"Unknown population-compartment morphology: {morphology}")
    prefix = f"{dynamics}-" + ("star-population-compartment-" if morphology == "star" else "population-compartment-")
    mode = prefix + ("json" if record else "no-record-json")
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / (
        f"cm_default_gpu_{dynamics}_{morphology}_population_compartment_"
        f"{n_neurons}_{n_added_compartments}_{recording_suffix}.json")
    subprocess.check_call([
        sys.executable,
        runner,
        mode,
        str(output_path),
        str(n_neurons),
        str(n_added_compartments),
        str(sample_neuron),
        str(sample_compartment),
    ])
    with open(output_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def run_gpu_active_compartment_cm_default(
        tmp_path, n_added_compartments, sample_compartment, record=True, morphology="chain", dynamics=None):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    if morphology not in ("chain", "star"):
        raise ValueError(f"Unknown compartment morphology: {morphology}")
    prefix = f"{dynamics}-" + ("star-" if morphology == "star" else "")
    mode = prefix + ("compartment-json" if record else "compartment-no-record-json")
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / (
        f"cm_default_gpu_{dynamics}_{morphology}_compartment_{n_added_compartments}_{recording_suffix}.json")
    subprocess.check_call([
        sys.executable,
        runner,
        mode,
        str(output_path),
        str(n_added_compartments),
        str(sample_compartment),
    ])
    with open(output_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def compare_population_traces(cpu, gpu, dynamics):
    for compartment in (0, 1):
        compare_compartment_traces(cpu, gpu, compartment, dynamics)


def compare_compartment_traces(cpu, gpu, compartment, dynamics):
    variable = f"v_comp{compartment}"
    atol = 2.0 if dynamics == "active" and compartment == 0 else 0.5
    compare_trace(cpu["traces"], gpu["traces"], variable, variable, atol=atol)


def compare_2d_traces(cpu, gpu, compartment, dynamics):
    compare_compartment_traces(cpu, gpu, compartment, dynamics)


def plot_population_benchmark(results, output_dir, variant, dynamics):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gpu_compartmental_{variant}_{dynamics}_population_benchmark.png")
    print(f"Writing GPU compartmental benchmark plot to {output_path}")

    n_neurons = np.asarray([result["n_neurons"] for result in results], dtype=int)
    cpu_runtimes = np.asarray([result["cpu_runtime"] for result in results], dtype=float)
    gpu_runtimes = np.asarray([result["gpu_runtime"] for result in results], dtype=float)
    cpu_no_record_runtimes = np.asarray(
        [result["cpu_no_record_runtime"] for result in results], dtype=float)
    gpu_no_record_runtimes = np.asarray([result["gpu_no_record_runtime"] for result in results], dtype=float)
    relative_runtimes = gpu_runtimes / cpu_runtimes
    relative_no_record_runtimes = gpu_no_record_runtimes / cpu_no_record_runtimes

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_neurons, relative_runtimes, marker="o", label=f"{variant} with recording")
    ax.plot(n_neurons, relative_no_record_runtimes, marker="s", label=f"{variant} without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("population size")
    ax.set_ylabel("relative runtime")
    ax.set_title(f"cm_default {variant} {dynamics} population benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_compartment_benchmark(results, output_dir, variant, morphology="chain", dynamics="active"):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental compartment benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"gpu_compartmental_{variant}_{dynamics}_{morphology}_compartment_benchmark.png"
    output_path = os.path.join(output_dir, output_filename)
    print(f"Writing GPU compartmental compartment benchmark plot to {output_path}")

    n_compartments = np.asarray([result["n_added_compartments"] for result in results], dtype=int)
    cpu_runtimes = np.asarray([result["cpu_runtime"] for result in results], dtype=float)
    gpu_runtimes = np.asarray([result["gpu_runtime"] for result in results], dtype=float)
    cpu_no_record_runtimes = np.asarray(
        [result["cpu_no_record_runtime"] for result in results], dtype=float)
    gpu_no_record_runtimes = np.asarray([result["gpu_no_record_runtime"] for result in results], dtype=float)
    relative_runtimes = gpu_runtimes / cpu_runtimes
    relative_no_record_runtimes = gpu_no_record_runtimes / cpu_no_record_runtimes

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_compartments, relative_runtimes, marker="o", label=f"{variant} with recording")
    ax.plot(n_compartments, relative_no_record_runtimes, marker="s", label=f"{variant} without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("added dendritic compartment count")
    ax.set_ylabel("relative runtime")
    ax.set_title(f"cm_default {variant} {dynamics} {morphology} compartment benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _load_benchmark_results(target_path, variant, dynamics, kind):
    json_path = os.path.join(
        target_path, f"gpu_compartmental_{variant}_{dynamics}_{kind}_benchmark.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _plot_all_variants(target_path, dynamics, kind, x_label):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental all-variants benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(target_path, exist_ok=True)
    output_path = os.path.join(
        target_path, f"gpu_compartmental_{dynamics}_{kind}_benchmark_all_variants.png")
    print(f"Writing GPU compartmental all-variants benchmark plot to {output_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in ("default", "fastexp"):
        results = _load_benchmark_results(target_path, variant, dynamics, kind)
        if results is None:
            continue
        x = np.asarray(
            [r["n_neurons"] if kind == "population" else r["n_added_compartments"] for r in results], dtype=int)
        gpu_runtimes = np.asarray([r["gpu_runtime"] for r in results], dtype=float)
        cpu_runtimes = np.asarray([r["cpu_runtime"] for r in results], dtype=float)
        relative_runtimes = gpu_runtimes / cpu_runtimes
        ax.plot(x, relative_runtimes, marker="o", label=f"{variant} with recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("relative runtime")
    ax.set_title(f"cm_default {dynamics} {kind} benchmark: default vs fastexp")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def pytest_sessionfinish(session, exit_status):
    # After every test has run, draw a combined default-vs-fastexp plot for each
    # (dynamics, kind) so the two variants are compared in a single figure.
    if not TEST_PLOTS:
        return
    target_path = os.path.join(TESTS_PATH, "target")
    for dynamics in ("active", "passive"):
        _plot_all_variants(target_path, dynamics, "population", "population size")
        _plot_all_variants(target_path, dynamics, "chain", "added dendritic compartment count")
        _plot_all_variants(target_path, dynamics, "star", "added dendritic compartment count")


def print_population_run_header(n_neurons, sample_neuron, dynamics):
    print(f"\n=== cm_default {dynamics} population benchmark ===")
    print(f"setup: {n_neurons} neuron(s), two compartments per neuron, one somatic AMPA receptor per neuron")
    print(f"sample: neuron {sample_neuron}, soma and dendrite voltage")


def print_compartment_run_header(n_added_compartments, sample_compartment, morphology="chain", dynamics="active"):
    print(f"\n=== cm_default {dynamics} {morphology} compartment-count benchmark ===")
    print(f"setup: one neuron, {n_added_compartments} added dendritic compartment(s), "
          f"{n_added_compartments + 1} total compartment(s)")
    if morphology == "chain":
        print("morphology: each dendrite is the child of the preceding compartment")
    else:
        print("morphology: every dendrite is a direct child of the soma")
    print("receptors: one AMPA receptor at soma, spike input connected to receptor port 0")
    print(f"sample: compartment {sample_compartment}, voltage")


def print_benchmark_results(title, size_label, results):
    print(f"\n=== {title} results ===")
    print(f"{size_label:>14}  {'sample':>8}  {'recording':>10}  "
          f"{'CPU NESTML [s]':>15}  {'NEST-GPU [s]':>14}  {'GPU/CPU':>10}")
    for result in results:
        if size_label == "neurons":
            size = result["n_neurons"]
            sample = result["sample_neuron"]
        else:
            size = result["n_added_compartments"]
            sample = result["sample_compartment"]
        for recording, cpu_key, gpu_key in (
                ("yes", "cpu_runtime", "gpu_runtime"),
                ("no", "cpu_no_record_runtime", "gpu_no_record_runtime")):
            cpu_runtime = result[cpu_key]
            gpu_runtime = result[gpu_key]
            ratio = gpu_runtime / cpu_runtime if cpu_runtime > 0 else float("inf")
            print(f"{size:14d}  {sample:8d}  {recording:>10}  {cpu_runtime:15.6f}  "
                  f"{gpu_runtime:14.6f}  {ratio:10.3f}")


def _run_population_1d(cpu_model, target_path, variant, dynamics, gpu_tmp_dir):
    rng = np.random.default_rng(BENCHMARK_RANDOM_SEED)
    benchmark_results = []
    for n_neurons in BENCHMARK_POPULATION_SIZES:
        sample_neuron = int(rng.integers(0, n_neurons))
        print_population_run_header(n_neurons, sample_neuron, dynamics)
        print(f"running single-precision CPU NESTML reference simulation ({variant})")
        cpu = run_cpu_active_population_cm_default(
            cpu_model, n_neurons, sample_neuron, dynamics=dynamics)
        print("running generated NEST-GPU simulation")
        gpu = run_gpu_active_population_cm_default(gpu_tmp_dir, n_neurons, sample_neuron, dynamics=dynamics)
        compare_population_traces(cpu, gpu, dynamics)
        print("running single-precision CPU NESTML simulation without recording")
        cpu_no_record = run_cpu_active_population_cm_default(
            cpu_model, n_neurons, sample_neuron, record=False, dynamics=dynamics)
        print("running generated NEST-GPU simulation without recording")
        gpu_no_record = run_gpu_active_population_cm_default(
            gpu_tmp_dir, n_neurons, sample_neuron, record=False, dynamics=dynamics)
        benchmark_results.append({
            "n_neurons": n_neurons,
            "sample_neuron": sample_neuron,
            "dynamics": dynamics,
            "cpu_runtime": cpu["runtime"],
            "gpu_runtime": gpu["runtime"],
            "cpu_no_record_runtime": cpu_no_record["runtime"],
            "gpu_no_record_runtime": gpu_no_record["runtime"],
        })
        print(f"finished population run with recording: cpu={cpu['runtime']:.6f}s, "
              f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / cpu['runtime']:.3f}")
        print(f"finished population run without recording: cpu={cpu_no_record['runtime']:.6f}s, "
              f"gpu={gpu_no_record['runtime']:.6f}s, "
              f"ratio={gpu_no_record['runtime'] / cpu_no_record['runtime']:.3f}")

    print_benchmark_results(f"cm_default {dynamics} population benchmark", "neurons", benchmark_results)
    plot_population_benchmark(benchmark_results, target_path, variant, dynamics)
    json_path = os.path.join(
        target_path, f"gpu_compartmental_{variant}_{dynamics}_population_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(benchmark_results, output_file, indent=2)
    return benchmark_results


def _run_compartment_1d(cpu_model, target_path, variant, dynamics, morphology, gpu_tmp_dir):
    benchmark_results = []
    for n_added_compartments in BENCHMARK_COMPARTMENT_SIZES:
        sample_compartment = n_added_compartments
        print_compartment_run_header(
            n_added_compartments, sample_compartment, morphology=morphology, dynamics=dynamics)
        print(f"running single-precision CPU NESTML reference simulation ({variant})")
        cpu = run_cpu_active_compartment_cm_default(
            cpu_model, n_added_compartments, sample_compartment, morphology=morphology, dynamics=dynamics)
        print("running generated NEST-GPU simulation")
        gpu = run_gpu_active_compartment_cm_default(
            gpu_tmp_dir, n_added_compartments, sample_compartment, morphology=morphology, dynamics=dynamics)
        compare_compartment_traces(cpu, gpu, sample_compartment, dynamics)
        print("running single-precision CPU NESTML simulation without recording")
        cpu_no_record = run_cpu_active_compartment_cm_default(
            cpu_model, n_added_compartments, sample_compartment, record=False, morphology=morphology, dynamics=dynamics)
        print("running generated NEST-GPU simulation without recording")
        gpu_no_record = run_gpu_active_compartment_cm_default(
            gpu_tmp_dir, n_added_compartments, sample_compartment, record=False, morphology=morphology, dynamics=dynamics)
        benchmark_results.append({
            "n_added_compartments": n_added_compartments,
            "n_compartments": n_added_compartments + 1,
            "sample_compartment": sample_compartment,
            "morphology": morphology,
            "dynamics": dynamics,
            "cpu_runtime": cpu["runtime"],
            "gpu_runtime": gpu["runtime"],
            "cpu_no_record_runtime": cpu_no_record["runtime"],
            "gpu_no_record_runtime": gpu_no_record["runtime"],
        })
        print(f"finished {morphology} compartment run with recording: cpu={cpu['runtime']:.6f}s, "
              f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / cpu['runtime']:.3f}")
        print(f"finished {morphology} compartment run without recording: "
              f"cpu={cpu_no_record['runtime']:.6f}s, gpu={gpu_no_record['runtime']:.6f}s, "
              f"ratio={gpu_no_record['runtime'] / cpu_no_record['runtime']:.3f}")

    print_benchmark_results(
        f"cm_default {dynamics} {morphology} compartment-count benchmark", "added_comp", benchmark_results)
    plot_compartment_benchmark(benchmark_results, target_path, variant, morphology=morphology, dynamics=dynamics)
    json_path = os.path.join(
        target_path, f"gpu_compartmental_{variant}_{dynamics}_{morphology}_compartment_benchmark.json")
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(benchmark_results, output_file, indent=2)
    return benchmark_results


def run_benchmark_1d(cpu_model, target_path, variant, dynamics, kind="population", gpu_tmp_dir=None):
    """Run one 1-D benchmark variant and return its results list.

    ``kind`` is "population" (neurons, 2 comps/neuron) or "chain"/"star" (added dendritic
    compartments, 1 neuron). ``target_path`` receives the results JSON and plots (kept in
    step with ``pytest_sessionfinish``); ``gpu_tmp_dir`` (default ``target_path``) holds
    the per-run GPU JSONs. The tree solver is fixed by the module build (``NESTML_GPU_CM_SOLVER``).
    """
    if gpu_tmp_dir is None:
        gpu_tmp_dir = target_path
    if kind == "population":
        return _run_population_1d(cpu_model, target_path, variant, dynamics, gpu_tmp_dir)
    if kind in ("chain", "star"):
        return _run_compartment_1d(cpu_model, target_path, variant, dynamics, kind, gpu_tmp_dir)
    raise ValueError(f"Unknown 1-D benchmark kind: {kind!r}")


def _run_2d_combo(cpu_model, output_dir, variant, dynamics, morphology, n_neurons, n_added_compartments,
                  sample_neuron, sample_compartment, gpu_tmp_dir):
    """One (neurons, comps) point: CPU reference + GPU (recorded and unrecorded), trace-checked."""
    print_compartment_run_header(n_added_compartments, sample_compartment, morphology=morphology, dynamics=dynamics)
    print(f"  {n_neurons:6d} neurons x {n_added_compartments:3d} comps  [{variant}]")
    print("running single-precision CPU NESTML reference simulation")
    cpu = run_cpu_active_compartment_population_cm_default(
        cpu_model, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        morphology=morphology, dynamics=dynamics)
    print("running generated NEST-GPU simulation")
    gpu = run_gpu_active_compartment_population_cm_default(
        gpu_tmp_dir, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        morphology=morphology, dynamics=dynamics)
    compare_2d_traces(cpu, gpu, sample_compartment, dynamics)
    print("running single-precision CPU NESTML simulation without recording")
    cpu_no_record = run_cpu_active_compartment_population_cm_default(
        cpu_model, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        morphology=morphology, dynamics=dynamics, record=False)
    print("running generated NEST-GPU simulation without recording")
    gpu_no_record = run_gpu_active_compartment_population_cm_default(
        gpu_tmp_dir, n_neurons, n_added_compartments, sample_neuron, sample_compartment,
        morphology=morphology, dynamics=dynamics, record=False)
    return {
        "n_neurons": n_neurons,
        "n_added_compartments": n_added_compartments,
        "n_compartments": n_added_compartments + 1,
        "sample_neuron": sample_neuron,
        "sample_compartment": sample_compartment,
        "morphology": morphology,
        "dynamics": dynamics,
        "variant": variant,
        "cpu_runtime": cpu["runtime"],
        "gpu_runtime": gpu["runtime"],
        "cpu_no_record_runtime": cpu_no_record["runtime"],
        "gpu_no_record_runtime": gpu_no_record["runtime"],
    }


def _plot_2d_benchmark(results, output_dir, variant, dynamics, morphology):
    if not TEST_PLOTS:
        print(f"Skipping 2-D benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"gpu_compartmental_{variant}_{dynamics}_{morphology}_2d_benchmark.png")
    neurons = sorted({r["n_neurons"] for r in results})
    comps = sorted({r["n_added_compartments"] for r in results})
    grid = {(r["n_neurons"], r["n_added_compartments"]): r["gpu_runtime"] for r in results}
    fig, ax = plt.subplots(figsize=(7, 4))
    for n_neurons in neurons:
        xs = [c for c in comps if (n_neurons, c) in grid]
        ys = [grid[(n_neurons, c)] for c in xs]
        ax.plot(xs, ys, marker="o", label=f"{n_neurons} neurons")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("added dendritic compartment count")
    ax.set_ylabel("NEST-GPU sim time [s]")
    ax.set_title(f"cm_default {variant} {dynamics} {morphology} 2-D benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _load_2d_results(output_dir, variant, dynamics, morphology):
    json_path = os.path.join(
        output_dir, f"gpu_compartmental_{variant}_{dynamics}_{morphology}_2d_benchmark.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def run_benchmark_2d(cpu_model, output_dir, variant, dynamics, morphologies=("chain", "star"),
                     neuron_sizes=(128, 256, 512, 1024, 2048),
                     comp_sizes=(8, 16, 32, 64, 128), sample_neuron=0, gpu_tmp_dir=None):
    """Neurons x compartments 2-D benchmark.

    For each ``morphology`` in ``morphologies`` and each ``neuron_sizes`` x ``comp_sizes``
    combo, runs a CPU reference and a NEST-GPU simulation (recorded and unrecorded),
    checks the traces, and returns a flat results list. Writes one results JSON + plot per
    morphology to ``output_dir``.
    """
    for morphology in morphologies:
        if morphology not in ("chain", "star"):
            raise ValueError(f"Unknown 2-D morphology: {morphology!r}")
    if gpu_tmp_dir is None:
        gpu_tmp_dir = output_dir
    all_results = []
    for morphology in morphologies:
        morphology_results = []
        for n_neurons in neuron_sizes:
            for n_added_compartments in comp_sizes:
                sample_neuron = min(sample_neuron, n_neurons - 1)
                sample_compartment = n_added_compartments
                print(f"\n=== cm_default {variant} {dynamics} {morphology} 2-D benchmark ===")
                result = _run_2d_combo(
                    cpu_model, output_dir, variant, dynamics, morphology,
                    n_neurons, n_added_compartments, sample_neuron, sample_compartment, gpu_tmp_dir)
                print(f"  finished: cpu={result['cpu_runtime']:.6f}s, "
                      f"gpu={result['gpu_runtime']:.6f}s, "
                      f"ratio={result['gpu_runtime'] / result['cpu_runtime']:.3f}")
                morphology_results.append(result)
        all_results.extend(morphology_results)

        print_benchmark_results(
            f"cm_default {variant} {dynamics} {morphology} 2-D benchmark", "neurons", morphology_results)
        _plot_2d_benchmark(morphology_results, output_dir, variant, dynamics, morphology)
        json_path = os.path.join(
            output_dir, f"gpu_compartmental_{variant}_{dynamics}_{morphology}_2d_benchmark.json")
        with open(json_path, "w", encoding="utf-8") as output_file:
            json.dump(morphology_results, output_file, indent=2)
    return all_results


class TestNESTGPUCompartmentalPopulationBenchmark:
    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_population_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_models, dynamics):
        run_benchmark_1d(
            benchmark_models["cpu"], benchmark_models["gpu_path"], benchmark_models["variant"],
            dynamics, kind="population", gpu_tmp_dir=tmp_path)

    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_compartment_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_models, dynamics):
        run_benchmark_1d(
            benchmark_models["cpu"], benchmark_models["gpu_path"], benchmark_models["variant"],
            dynamics, kind="chain", gpu_tmp_dir=tmp_path)

    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_star_compartment_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_models, dynamics):
        run_benchmark_1d(
            benchmark_models["cpu"], benchmark_models["gpu_path"], benchmark_models["variant"],
            dynamics, kind="star", gpu_tmp_dir=tmp_path)

    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_2d_compartment_population_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_models, dynamics):
        run_benchmark_2d(
            benchmark_models["cpu"], benchmark_models["gpu_path"], benchmark_models["variant"],
            dynamics, morphologies=("chain", "star"), gpu_tmp_dir=tmp_path)


def _parse_sizes(text):
    """Parse a comma-separated list of ints (used for the 2-D grid flags)."""
    return tuple(int(item) for item in text.split(",") if item.strip())


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Run the NEST-GPU compartmental benchmark (1-D or 2-D).")
    parser.add_argument("mode", choices=["1d", "2d"], help="which benchmark to run")
    parser.add_argument(
        "--variant", default=os.environ.get(VARIANT_ENV, DEFAULT_VARIANT),
        choices=["default", "fastexp", "all"],
        help="propagator variant(s); 'all' runs both (default: from $%s or 'default')" % VARIANT_ENV)
    parser.add_argument(
        "--dynamics", default="passive", choices=["active", "passive"],
        help="dynamics variant (default: passive)")
    parser.add_argument(
        "--morphology", default=None,
        help="run only this morphology (chain|star|population); default: all for the mode")
    parser.add_argument(
        "--output-dir", default=os.path.join(TESTS_PATH, "target"),
        help="directory for results JSON and plots (default: tests/nest_gpu_compartmental_tests/target)")
    parser.add_argument(
        "--neuron-sizes", type=_parse_sizes, default=None,
        help="comma-separated neuron counts for the 2-D grid (default: 128,256,512,1024,2048)")
    parser.add_argument(
        "--comp-sizes", type=_parse_sizes, default=None,
        help="comma-separated added-compartment counts for the 2-D grid (default: 8,16,32,64,128)")
    parser.add_argument(
        "--sample-neuron", type=int, default=0,
        help="neuron index to record from (default: 0)")
    return parser


def _variants_to_run(variant):
    if variant == "all":
        return ["default", "fastexp"]
    return [variant]


def main(argv=None):
    args = _build_cli_parser().parse_args(argv)
    solver = os.environ.get(SOLVER_ENV, DEFAULT_SOLVER)

    for variant in _variants_to_run(args.variant):
        target_path = _build_gpu_target(solver, variant)
        cpu_model = _build_cpu_model(variant)
        print(f"\n=== benchmark {args.mode}: variant={variant} solver={solver} dynamics={args.dynamics} ===")

        if args.mode == "1d":
            if args.morphology:
                run_benchmark_1d(
                    cpu_model, target_path, variant, args.dynamics, kind=args.morphology)
            else:
                for kind in ("population", "chain", "star"):
                    run_benchmark_1d(
                        cpu_model, target_path, variant, args.dynamics, kind=kind)
        else:  # 2d
            morphologies = (args.morphology,) if args.morphology else ("chain", "star")
            run_benchmark_2d(
                cpu_model, target_path, variant, args.dynamics,
                morphologies=morphologies,
                neuron_sizes=args.neuron_sizes or (128, 256, 512, 1024, 2048),
                comp_sizes=args.comp_sizes or (8, 16, 32, 64, 128),
                sample_neuron=args.sample_neuron)


if __name__ == "__main__":
    main()
