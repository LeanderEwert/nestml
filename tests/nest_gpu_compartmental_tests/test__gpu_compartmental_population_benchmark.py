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

"""Existing population/chain/star sweeps, each with active and passive dynamics.

Both variants use somatic AMPA only and identical voltage recording. Run in
~/.venvs/nestml-gpu; pytest -k active or -k passive selects a dynamics variant.
Runtime includes setup and one simulation, excluding trace retrieval.
"""

import json
import os
import subprocess
import sys
import time

import nest
import numpy as np
import pytest

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

if TEST_PLOTS:
    import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def benchmark_target_path():
    target_path = os.path.join(TESTS_PATH, "target")
    nest_gpu_path = os.environ.get("NEST_GPU", os.getcwd())
    assert os.path.isdir(os.path.join(nest_gpu_path, "src"))

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        os.makedirs(target_path, exist_ok=True)
        print(f"Skipping cm_default_nestml rebuild because {SKIP_REBUILD_ENV} is set")
    else:
        generate_gpu_default_model(target_path, codegen_opts={
            "gpu_tree_solver": "r_edd",
            "gpu_tree_solver_chain_length": 3,
            "gpu_tree_solver_base_size": 32,
        })

    return target_path


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


def plot_population_benchmark(results, output_dir, dynamics):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gpu_compartmental_{dynamics}_population_benchmark.png")
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
    ax.plot(n_neurons, relative_runtimes, marker="o", label="with recording")
    ax.plot(n_neurons, relative_no_record_runtimes, marker="s", label="without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("population size")
    ax.set_ylabel("relative runtime")
    ax.set_title(f"cm_default {dynamics} population benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_compartment_benchmark(results, output_dir, morphology="chain", dynamics="active"):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental compartment benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"gpu_compartmental_{dynamics}_{morphology}_compartment_benchmark.png"
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
    ax.plot(n_compartments, relative_runtimes, marker="o", label="with recording")
    ax.plot(n_compartments, relative_no_record_runtimes, marker="s", label="without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("added dendritic compartment count")
    ax.set_ylabel("relative runtime")
    ax.set_title(f"cm_default {dynamics} {morphology} compartment benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


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


class TestNESTGPUCompartmentalPopulationBenchmark:
    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_population_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_target_path, cpu_single_precision_cm_default, dynamics):
        rng = np.random.default_rng(BENCHMARK_RANDOM_SEED)
        benchmark_results = []
        for n_neurons in BENCHMARK_POPULATION_SIZES:
            sample_neuron = int(rng.integers(0, n_neurons))
            print_population_run_header(n_neurons, sample_neuron, dynamics)
            print("running single-precision CPU NESTML reference simulation")
            cpu = run_cpu_active_population_cm_default(
                cpu_single_precision_cm_default, n_neurons, sample_neuron, dynamics=dynamics)
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_population_cm_default(tmp_path, n_neurons, sample_neuron, dynamics=dynamics)
            compare_population_traces(cpu, gpu, dynamics)
            print("running single-precision CPU NESTML simulation without recording")
            cpu_no_record = run_cpu_active_population_cm_default(
                cpu_single_precision_cm_default, n_neurons, sample_neuron, record=False, dynamics=dynamics)
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_population_cm_default(
                tmp_path, n_neurons, sample_neuron, record=False, dynamics=dynamics)
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
        plot_population_benchmark(benchmark_results, benchmark_target_path, dynamics)
        with open(os.path.join(benchmark_target_path, f"gpu_compartmental_{dynamics}_population_benchmark.json"), "w",
                  encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)

    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_compartment_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_target_path, cpu_single_precision_cm_default, dynamics):
        benchmark_results = []
        for n_added_compartments in BENCHMARK_COMPARTMENT_SIZES:
            sample_compartment = n_added_compartments
            print_compartment_run_header(n_added_compartments, sample_compartment, dynamics=dynamics)
            print("running single-precision CPU NESTML reference simulation")
            cpu = run_cpu_active_compartment_cm_default(
                cpu_single_precision_cm_default, n_added_compartments, sample_compartment, dynamics=dynamics)
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_compartment_cm_default(tmp_path, n_added_compartments, sample_compartment, dynamics=dynamics)
            compare_compartment_traces(cpu, gpu, sample_compartment, dynamics)
            print("running single-precision CPU NESTML simulation without recording")
            cpu_no_record = run_cpu_active_compartment_cm_default(
                cpu_single_precision_cm_default, n_added_compartments, sample_compartment, record=False, dynamics=dynamics)
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, record=False, dynamics=dynamics)
            benchmark_results.append({
                "n_added_compartments": n_added_compartments,
                "n_compartments": n_added_compartments + 1,
                "sample_compartment": sample_compartment,
                "dynamics": dynamics,
                "cpu_runtime": cpu["runtime"],
                "gpu_runtime": gpu["runtime"],
                "cpu_no_record_runtime": cpu_no_record["runtime"],
                "gpu_no_record_runtime": gpu_no_record["runtime"],
            })
            print(f"finished compartment run with recording: cpu={cpu['runtime']:.6f}s, "
                  f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / cpu['runtime']:.3f}")
            print(f"finished compartment run without recording: cpu={cpu_no_record['runtime']:.6f}s, "
                  f"gpu={gpu_no_record['runtime']:.6f}s, "
                  f"ratio={gpu_no_record['runtime'] / cpu_no_record['runtime']:.3f}")

        print_benchmark_results(f"cm_default {dynamics} chain compartment-count benchmark", "added_comp", benchmark_results)
        plot_compartment_benchmark(benchmark_results, benchmark_target_path, morphology="chain", dynamics=dynamics)
        with open(os.path.join(benchmark_target_path, f"gpu_compartmental_{dynamics}_chain_compartment_benchmark.json"), "w",
                  encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)

    @pytest.mark.parametrize("dynamics", ["active", "passive"])
    def test_cm_default_star_compartment_benchmark_against_single_precision_cpu_nestml(
            self, tmp_path, benchmark_target_path, cpu_single_precision_cm_default, dynamics):
        benchmark_results = []
        for n_added_compartments in BENCHMARK_COMPARTMENT_SIZES:
            sample_compartment = n_added_compartments
            print_compartment_run_header(n_added_compartments, sample_compartment, morphology="star", dynamics=dynamics)
            print("running single-precision CPU NESTML reference simulation")
            cpu = run_cpu_active_compartment_cm_default(
                cpu_single_precision_cm_default, n_added_compartments,
                sample_compartment, morphology="star", dynamics=dynamics)
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, morphology="star", dynamics=dynamics)
            compare_compartment_traces(cpu, gpu, sample_compartment, dynamics)
            print("running single-precision CPU NESTML simulation without recording")
            cpu_no_record = run_cpu_active_compartment_cm_default(
                cpu_single_precision_cm_default, n_added_compartments,
                sample_compartment, record=False, morphology="star", dynamics=dynamics)
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, record=False, morphology="star", dynamics=dynamics)
            benchmark_results.append({
                "n_added_compartments": n_added_compartments,
                "n_compartments": n_added_compartments + 1,
                "sample_compartment": sample_compartment,
                "morphology": "star",
                "dynamics": dynamics,
                "cpu_runtime": cpu["runtime"],
                "gpu_runtime": gpu["runtime"],
                "cpu_no_record_runtime": cpu_no_record["runtime"],
                "gpu_no_record_runtime": gpu_no_record["runtime"],
            })
            print(f"finished star compartment run with recording: cpu={cpu['runtime']:.6f}s, "
                  f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / cpu['runtime']:.3f}")
            print(f"finished star compartment run without recording: "
                  f"cpu={cpu_no_record['runtime']:.6f}s, gpu={gpu_no_record['runtime']:.6f}s, "
                  f"ratio={gpu_no_record['runtime'] / cpu_no_record['runtime']:.3f}")

        print_benchmark_results(
            f"cm_default {dynamics} star compartment-count benchmark", "added_comp", benchmark_results)
        plot_compartment_benchmark(benchmark_results, benchmark_target_path, morphology="star", dynamics=dynamics)
        with open(os.path.join(
                benchmark_target_path, f"gpu_compartmental_{dynamics}_star_compartment_benchmark.json"), "w",
                encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)
