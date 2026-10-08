# -*- coding: utf-8 -*-
#
# This file is part of NEST, distributed under the GNU General Public License
# version 2 or later. See the file LICENSE for details.
"""Population sweep of the standard CPU and GPU iaf_psc_exp models.

Run with pytest -s. Configuration and timing definitions are documented in
README_iaf_psc_exp_benchmark.md. No code generation or rebuild is required.
"""

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pytest

from iaf_psc_exp_benchmark_runner import PARAMETERS, PSC_DELAY, PSC_SPIKE_TIMES, VALIDATION_DURATION


HERE = Path(__file__).resolve().parent
PREFIX = "NESTML_IAF_BENCHMARK_"
SCOPES = ("setup_first_s", "warm_simulation_s")
SCOPE_LABELS = ("Setup + first simulation", "Warmed simulation")


def benchmark_config():
    sizes_text = os.environ.get(PREFIX + "SIZES")
    sizes = sorted(set(int(value) for value in sizes_text.split(","))) if sizes_text else [2**i for i in range(21)]
    config = {
        "sizes": sizes, "repeats": int(os.environ.get(PREFIX + "REPEATS", "3")),
        "threads": int(os.environ.get(PREFIX + "THREADS", "1")),
        "duration_ms": float(os.environ.get(PREFIX + "DURATION_MS", "1000")),
        "dt_ms": float(os.environ.get(PREFIX + "DT_MS", "0.1")),
        "timeout_s": float(os.environ.get(PREFIX + "TIMEOUT_S", "600")),
        "gpu_spike_buffer_algo": int(os.environ.get(PREFIX + "GPU_SPIKE_BUFFER_ALGO", "1")),
        "seed": 12345,
    }
    assert sizes and min(sizes) > 0, "Population sizes must be positive"
    assert config["repeats"] > 0 and config["threads"] > 0
    assert config["gpu_spike_buffer_algo"] in (0, 1)
    dt = config["dt_ms"]
    assert np.isfinite(dt) and 0 < dt <= 0.5, "Require 0 < dt <= 0.5 ms"
    assert np.isfinite(config["duration_ms"]) and config["duration_ms"] >= dt
    assert np.isfinite(config["timeout_s"]) and config["timeout_s"] > 0
    for value in [config["duration_ms"], VALIDATION_DURATION, PSC_DELAY, PARAMETERS["t_ref"], *PSC_SPIKE_TIMES]:
        assert np.isclose(value / dt, round(value / dt), rtol=0, atol=1e-6), "Times must lie on the time grid"
    return config


def run_worker(tmp_path, config, backend, n_neurons=1, sample=0, record=False, validation=False):
    output = tmp_path / f"iaf_{backend}.json"
    output.unlink(missing_ok=True)
    command = [sys.executable, str(HERE / "iaf_psc_exp_benchmark_runner.py"), backend, str(output),
               "--neurons", str(n_neurons), "--sample", str(sample),
               "--threads", str(config["threads"]), "--dt", str(config["dt_ms"]),
               "--gpu-spike-buffer-algo", str(config["gpu_spike_buffer_algo"]),
               "--duration", str(config["duration_ms"])]
    if record:
        command.append("--record")
    if validation:
        command.append("--validation")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=config["timeout_s"], check=False)
    assert completed.returncode == 0, (
        f"{backend} worker failed ({completed.returncode}), N={n_neurons}, record={record}\n"
        f"{completed.stdout}\n{completed.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))


def compare_traces(cpu, gpu, dt, duration, validation=False):
    """Compare real common grid samples, without interpolation or fitted shifts."""
    assert len(cpu["traces"]) == len(gpu["traces"]) > 0
    max_errors = {}
    for c_trace, g_trace in zip(cpu["traces"], gpu["traces"]):
        assert c_trace["neuron"] == g_trace["neuron"]
        ticks = []
        for trace in (c_trace, g_trace):
            times = np.asarray(trace["times"])
            assert np.all(np.isfinite(times)) and len(times) > 0
            grid = np.rint(times / dt).astype(np.int64)
            np.testing.assert_allclose(times, grid * dt, rtol=0, atol=max(1e-5, duration * 1e-7))
            assert np.all(np.diff(grid) == 1), "Missing or duplicated recording steps"
            ticks.append(grid)
        common, ci, gi = np.intersect1d(*ticks, return_indices=True)
        # CPU recorders can retain the final delay interval; GPU also records t=0.
        assert len(common) >= max(1, int(0.95 * duration / dt)), "Insufficient overlapping recording"
        for variable in ("V_m", "I_syn_ex", "I_syn_in") if validation else ("V_m",):
            c_values = np.asarray(c_trace[variable])[ci]
            g_values = np.asarray(g_trace[variable])[gi]
            assert np.all(np.isfinite(c_values)) and np.all(np.isfinite(g_values))
            mask = np.ones(len(common), dtype=bool)
            if validation and variable.startswith("I_syn"):
                # GPU records before delivery, CPU after delivery: exclude only
                # the three discontinuity samples, then compare the full decay.
                arrival_ticks = np.rint((np.asarray(PSC_SPIKE_TIMES) + PSC_DELAY) / dt).astype(int)
                mask = ~np.isin(common, arrival_ticks)
            atol = 0.002 if variable == "V_m" else 0.01
            np.testing.assert_allclose(c_values[mask], g_values[mask], rtol=0, atol=atol,
                                       err_msg=f"Neuron {c_trace['neuron']}, {variable}")
            max_errors[f"neuron_{c_trace['neuron']}_{variable}"] = float(np.max(np.abs(c_values[mask] - g_values[mask])))
    return max_errors


def validate_models(cpu, gpu, dt):
    errors = compare_traces(cpu, gpu, dt, VALIDATION_DURATION, validation=True)
    assert len(cpu["spike_times"][0]) >= 2, "DC validation must exercise repeated spiking"
    for c_times, g_times in zip(cpu["spike_times"], gpu["spike_times"]):
        assert len(c_times) == len(g_times), "Spike counts differ"
        np.testing.assert_allclose(c_times, g_times, rtol=0, atol=dt + 1e-5)
    for result in (cpu, gpu):
        assert result["spike_times"][1:] == [[], []], "PSC probes should stay subthreshold"
        assert max(result["traces"][1]["I_syn_ex"]) > 100
        assert min(result["traces"][2]["I_syn_in"]) < -100
        assert max(result["traces"][1]["V_m"]) > PARAMETERS["E_L"] + 0.1
        assert min(result["traces"][2]["V_m"]) < PARAMETERS["E_L"] - 0.1
    return {"max_absolute_errors": errors, "spike_counts": [len(t) for t in cpu["spike_times"]],
            "voltage_atol_mV": 0.002, "current_atol_pA": 0.01, "spike_atol_ms": dt + 1e-5,
            "gpu_generator_advance_ms": dt,
            "psc_current_discontinuity_samples_excluded_ms": [t + PSC_DELAY for t in PSC_SPIKE_TIMES]}


def hardware_metadata():
    result = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
              "logical_cpus": os.cpu_count(), "cpu": platform.processor(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        result["cpu"] = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                              if line.startswith("model name")), result["cpu"])
    if hasattr(os, "sched_getaffinity"):
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
        result["nvidia_smi"] = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        result["nvidia_smi"] = str(error)
    return result


def summarize(runs):
    rows = []
    for size in sorted({run["n_neurons"] for run in runs}):
        for record in (False, True):
            subset = [run for run in runs if run["n_neurons"] == size and run["recording_enabled"] == record]
            if not subset:
                continue
            row = {"n_neurons": size, "recording_enabled": record, "repeats": len(subset)}
            for scope in SCOPES:
                cpu = np.asarray([run["cpu"][scope] for run in subset])
                gpu = np.asarray([run["gpu"][scope] for run in subset])
                row[scope] = {"cpu_median_s": float(np.median(cpu)), "gpu_median_s": float(np.median(gpu)),
                              "cpu_min_s": float(cpu.min()), "cpu_max_s": float(cpu.max()),
                              "gpu_min_s": float(gpu.min()), "gpu_max_s": float(gpu.max()),
                              "gpu_over_cpu": float(np.median(gpu) / np.median(cpu)),
                              "ratio_min": float(gpu.min() / cpu.max()),
                              "ratio_max": float(gpu.max() / cpu.min())}
            rows.append(row)
    return rows


def find_crossovers(summary):
    results = []
    for record in (False, True):
        rows = [row for row in summary if row["recording_enabled"] == record]
        for scope in SCOPES:
            brackets = []
            for left, right in zip(rows, rows[1:]):
                if (left[scope]["gpu_over_cpu"] < 1) != (right[scope]["gpu_over_cpu"] < 1):
                    brackets.append({"lower_n": left["n_neurons"], "upper_n": right["n_neurons"],
                                     "gpu_faster_at_upper": right[scope]["gpu_over_cpu"] < 1})
            results.append({"recording_enabled": record, "scope": scope, "brackets": brackets,
                            "gpu_faster_at_smallest": bool(rows and rows[0][scope]["gpu_over_cpu"] < 1),
                            "exact_equal_sizes": [row["n_neurons"] for row in rows if row[scope]["gpu_over_cpu"] == 1]})
    return results


def plot_results(summary, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Skipping plots: {error}")
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, (scope, label) in enumerate(zip(SCOPES, SCOPE_LABELS)):
        runtime_ax, ratio_ax = axes[:, column]
        for record, marker in ((False, "o"), (True, "s")):
            rows = [row for row in summary if row["recording_enabled"] == record]
            sizes = [row["n_neurons"] for row in rows]
            recording = "recorded" if record else "unrecorded"
            for backend in ("cpu", "gpu"):
                line, = runtime_ax.plot(sizes, [row[scope][backend + "_median_s"] for row in rows],
                                        marker=marker, label=f"{backend.upper()} {recording}")
                runtime_ax.fill_between(sizes, [row[scope][backend + "_min_s"] for row in rows],
                                        [row[scope][backend + "_max_s"] for row in rows],
                                        color=line.get_color(), alpha=0.15)
            line, = ratio_ax.plot(sizes, [row[scope]["gpu_over_cpu"] for row in rows],
                                  marker=marker, label=recording)
            ratio_ax.fill_between(sizes, [row[scope]["ratio_min"] for row in rows],
                                  [row[scope]["ratio_max"] for row in rows], color=line.get_color(), alpha=0.15)
        runtime_ax.set_title(label)
        runtime_ax.set_ylabel("Wall-clock runtime [s]")
        ratio_ax.set_ylabel("GPU / CPU runtime (below 1: GPU faster)")
        ratio_ax.axhline(1, color="grey", linestyle="--")
        for ax in (runtime_ax, ratio_ax):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("Population size")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
    fig.suptitle("Built-in iaf_psc_exp: medians and observed ranges")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"iaf_psc_exp_population_benchmark.{suffix}")
    plt.close(fig)


def test_iaf_psc_exp_population_benchmark(tmp_path):
    for module in ("nest", "nestgpu"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for the CPU/GPU benchmark")
    config = benchmark_config()
    output_dir = Path(os.environ.get(PREFIX + "OUTPUT_DIR", str(HERE / "target" / "iaf_benchmark")))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "model": "iaf_psc_exp", "parameters": PARAMETERS,
              "hardware": hardware_metadata(), "complete": False, "runs": [],
              "timing_notes": "Imports/kernel initialization excluded; first simulation warms up the second. "
                              "GPU timings end with cudaDeviceSynchronize. Retrieval/JSON excluded. "
                              "Recorded runs sample one V_m; unrecorded runs have no recorders."}
    path = output_dir / "iaf_psc_exp_population_benchmark.json"

    def save_report():
        report["summary"] = summarize(report["runs"])
        report["crossovers"] = find_crossovers(report["summary"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    save_report()
    print("Validating DC spikes and excitatory/inhibitory PSC responses", flush=True)
    cpu = run_worker(tmp_path, config, "cpu", validation=True)
    gpu = run_worker(tmp_path, config, "gpu", validation=True)
    report["backends"] = {"cpu": cpu["metadata"], "gpu": gpu["metadata"]}
    report["validation"] = validate_models(cpu, gpu, config["dt_ms"])
    save_report()
    rng = np.random.default_rng(config["seed"])
    for size in config["sizes"]:
        sample = int(rng.integers(size))
        for record in (False, True):
            for repeat in range(config["repeats"]):
                pair = {}
                # Alternate order to reduce systematic drift; never benchmark concurrently.
                for backend in (("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")):
                    pair[backend] = run_worker(tmp_path, config, backend, size, sample, record)
                errors = compare_traces(pair["cpu"], pair["gpu"], config["dt_ms"],
                                        2 * config["duration_ms"]) if record else None
                for result in pair.values():
                    for scope in (*SCOPES, "setup_s", "first_simulation_s", "retrieval_s"):
                        assert np.isfinite(result[scope]) and result[scope] > 0, f"Invalid timer: {scope}"
                    result.pop("traces")
                    result.pop("spike_times")
                    result.pop("metadata")
                report["runs"].append({"n_neurons": size, "sample_neuron": sample,
                                       "recording_enabled": record, "repeat": repeat,
                                       "trace_max_errors": errors, **pair})
                save_report()
                print(f"N={size:8d} record={record!s:5s} repeat={repeat + 1}: "
                      f"warm CPU={pair['cpu']['warm_simulation_s']:.6f}s "
                      f"GPU={pair['gpu']['warm_simulation_s']:.6f}s "
                      f"GPU/CPU={pair['gpu']['warm_simulation_s'] / pair['cpu']['warm_simulation_s']:.3f}", flush=True)
    report["complete"] = True
    save_report()
    plot_results(report["summary"], output_dir)
    for crossover in report["crossovers"]:
        label = f"{crossover['scope']}, recording={crossover['recording_enabled']}"
        if crossover["brackets"]:
            print(f"Crossover brackets ({label}): {crossover['brackets']}")
        elif crossover["exact_equal_sizes"]:
            print(f"Equal median runtimes ({label}): N={crossover['exact_equal_sizes']}")
        else:
            faster = "GPU" if crossover["gpu_faster_at_smallest"] else "CPU"
            print(f"No crossover observed ({label}); {faster} faster across tested sizes")
    print(f"Benchmark results: {path}")
