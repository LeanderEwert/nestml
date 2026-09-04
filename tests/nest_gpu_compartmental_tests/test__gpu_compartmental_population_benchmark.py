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
    DEND_PARAMS_ACTIVE,
    DEND_PARAMS_PASSIVE,
    DT,
    NEST_RECORDABLES,
    PLOT_IMPORT_ERROR,
    SIM_TIME,
    TEST_PLOTS,
    compare_conductance,
    compare_trace,
    configure_native_neuron,
    generate_gpu_default_model,
    SOMA_PARAMS,
)


BENCHMARK_POPULATION_SIZES = [2 ** i for i in range(6)]
BENCHMARK_COMPARTMENT_SIZES = [2 ** i for i in range(6)]
BENCHMARK_RANDOM_SEED = 12345
COMPARTMENT_BENCHMARK_SPIKE_TIMES = [10.0, 13.0, 16.0]
SKIP_REBUILD_ENV = "NESTML_GPU_CM_SKIP_REBUILD"

SOMA_PARAMS_PASSIVE = {
    "C_m": SOMA_PARAMS["C_m"],
    "g_C": SOMA_PARAMS["g_C"],
    "g_L": SOMA_PARAMS["g_L"],
    "e_L": SOMA_PARAMS["e_L"],
}


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
        generate_gpu_default_model(target_path)

    return target_path


def run_native_active_population_cm_default(n_neurons, sample_neuron, record=True):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    t_start = time.perf_counter()
    neurons = nest.Create("cm_default", n_neurons)
    configure_native_neuron(neurons, DEND_PARAMS_ACTIVE)

    sg_soma = nest.Create("spike_generator", n_neurons, {"spike_times": [10.0, 13.0, 16.0]})
    sg_dend = nest.Create("spike_generator", n_neurons, {"spike_times": [70.0, 73.0, 76.0]})

    nest.Connect(sg_soma, neurons, conn_spec={"rule": "one_to_one"}, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})
    nest.Connect(sg_dend, neurons, conn_spec={"rule": "one_to_one"}, syn_spec={
        "synapse_model": "static_synapse", "weight": 2.0, "delay": 0.5, "receptor_type": 1})

    if record:
        multimeter = nest.Create("multimeter", 1, {"record_from": NEST_RECORDABLES, "interval": DT})
        nest.Connect(multimeter, neurons[sample_neuron])

    nest.Simulate(SIM_TIME)
    runtime = time.perf_counter() - t_start

    result = {
        "n_neurons": n_neurons,
        "sample_neuron": sample_neuron,
        "recording_enabled": record,
        "runtime": runtime,
    }
    if record:
        result["active"] = nest.GetStatus(multimeter, "events")[0]
    return result


def run_gpu_active_population_cm_default(tmp_path, n_neurons, sample_neuron, record=True):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    mode = "active-population-json" if record else "active-population-no-record-json"
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / f"cm_default_gpu_population_{n_neurons}_{recording_suffix}.json"
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


def native_active_recordables_for_compartment(compartment):
    return [
        f"v_comp{compartment}",
        f"m_Na_{compartment}",
        f"h_Na_{compartment}",
        f"n_K_{compartment}",
    ]


def configure_native_active_compartment_chain(neuron, n_added_compartments):
    n_compartments = n_added_compartments + 1
    neuron.compartments = [
        {"parent_idx": -1, "params": SOMA_PARAMS},
        *[
            {"parent_idx": compartment - 1, "params": DEND_PARAMS_ACTIVE}
            for compartment in range(1, n_compartments)
        ],
    ]
    neuron.V_th = -50.0
    neuron.receptors = [
        {"comp_idx": 0, "receptor_type": "AMPA_NMDA"},
    ]


def configure_native_passive_compartment_star(neuron, n_added_compartments):
    neuron.compartments = [
        {"parent_idx": -1, "params": SOMA_PARAMS_PASSIVE},
        *[
            {"parent_idx": 0, "params": DEND_PARAMS_PASSIVE}
            for _ in range(n_added_compartments)
        ],
    ]
    neuron.V_th = -50.0
    neuron.receptors = [
        {"comp_idx": 0, "receptor_type": "AMPA"},
    ]


def run_native_active_compartment_cm_default(
        n_added_compartments, sample_compartment, record=True, morphology="chain"):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    recordables = ([f"v_comp{sample_compartment}"] if morphology == "star"
                   else native_active_recordables_for_compartment(sample_compartment))

    t_start = time.perf_counter()
    neuron = nest.Create("cm_default")
    if morphology == "chain":
        configure_native_active_compartment_chain(neuron, n_added_compartments)
    elif morphology == "star":
        configure_native_passive_compartment_star(neuron, n_added_compartments)
    else:
        raise ValueError(f"Unknown compartment morphology: {morphology}")

    spike_generator = nest.Create("spike_generator", 1, {"spike_times": COMPARTMENT_BENCHMARK_SPIKE_TIMES})

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
        "runtime": runtime,
    }
    if record:
        result["active"] = nest.GetStatus(multimeter, "events")[0]
    return result


def run_gpu_active_compartment_cm_default(
        tmp_path, n_added_compartments, sample_compartment, record=True, morphology="chain"):
    runner = os.path.join(TESTS_PATH, "gpu_compartmental_model_runner.py")
    if morphology == "chain":
        mode = "active-compartment-json" if record else "active-compartment-no-record-json"
    elif morphology == "star":
        mode = "passive-star-compartment-json" if record else "passive-star-compartment-no-record-json"
    else:
        raise ValueError(f"Unknown compartment morphology: {morphology}")
    recording_suffix = "recorded" if record else "unrecorded"
    output_path = tmp_path / (
        f"cm_default_gpu_{morphology}_compartment_{n_added_compartments}_{recording_suffix}.json")
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


def compare_active_default_traces(native, gpu):
    compare_trace(native["active"], gpu["active"], "v_comp0", "v_comp0", atol=2.0)
    compare_trace(native["active"], gpu["active"], "v_comp1", "v_comp1", atol=0.5)
    compare_trace(native["active"], gpu["active"], "m_Na_0", "m_Na0", atol=0.02)
    compare_trace(native["active"], gpu["active"], "h_Na_0", "h_Na0", atol=0.01)
    compare_trace(native["active"], gpu["active"], "n_K_0", "n_K0", atol=0.01)
    compare_trace(native["active"], gpu["active"], "m_Na_1", "m_Na1", atol=0.02)
    compare_trace(native["active"], gpu["active"], "h_Na_1", "h_Na1", atol=0.01)
    compare_trace(native["active"], gpu["active"], "n_K_1", "n_K1", atol=0.01)
    compare_conductance(native["active"], gpu["active"], "g_r_AN_AMPA_1", "g_d_AN_AMPA_1", "g_AN_AMPA1",
                        atol=0.03)
    compare_conductance(native["active"], gpu["active"], "g_r_AN_NMDA_1", "g_d_AN_NMDA_1", "g_AN_NMDA1",
                        atol=0.03)


def compare_active_compartment_default_traces(native, gpu, compartment):
    compare_trace(native["active"], gpu["active"], f"v_comp{compartment}", f"v_comp{compartment}", atol=2.0)
    compare_trace(native["active"], gpu["active"], f"m_Na_{compartment}", f"m_Na{compartment}", atol=0.02)
    compare_trace(native["active"], gpu["active"], f"h_Na_{compartment}", f"h_Na{compartment}", atol=0.01)
    compare_trace(native["active"], gpu["active"], f"n_K_{compartment}", f"n_K{compartment}", atol=0.01)


def plot_population_benchmark(results, output_dir):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gpu_compartmental_population_benchmark.png")
    print(f"Writing GPU compartmental benchmark plot to {output_path}")

    n_neurons = np.asarray([result["n_neurons"] for result in results], dtype=int)
    native_runtimes = np.asarray([result["native_runtime"] for result in results], dtype=float)
    gpu_runtimes = np.asarray([result["gpu_runtime"] for result in results], dtype=float)
    native_no_record_runtimes = np.asarray(
        [result["native_no_record_runtime"] for result in results], dtype=float)
    gpu_no_record_runtimes = np.asarray([result["gpu_no_record_runtime"] for result in results], dtype=float)
    relative_runtimes = gpu_runtimes / native_runtimes
    relative_no_record_runtimes = gpu_no_record_runtimes / native_no_record_runtimes

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_neurons, relative_runtimes, marker="o", label="with recording")
    ax.plot(n_neurons, relative_no_record_runtimes, marker="s", label="without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("population size")
    ax.set_ylabel("relative runtime")
    ax.set_title("cm_default active population benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_compartment_benchmark(results, output_dir, morphology="chain"):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental compartment benchmark plot: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_filename = ("gpu_compartmental_compartment_benchmark.png" if morphology == "chain"
                       else f"gpu_compartmental_{morphology}_compartment_benchmark.png")
    output_path = os.path.join(output_dir, output_filename)
    print(f"Writing GPU compartmental compartment benchmark plot to {output_path}")

    n_compartments = np.asarray([result["n_added_compartments"] for result in results], dtype=int)
    native_runtimes = np.asarray([result["native_runtime"] for result in results], dtype=float)
    gpu_runtimes = np.asarray([result["gpu_runtime"] for result in results], dtype=float)
    native_no_record_runtimes = np.asarray(
        [result["native_no_record_runtime"] for result in results], dtype=float)
    gpu_no_record_runtimes = np.asarray([result["gpu_no_record_runtime"] for result in results], dtype=float)
    relative_runtimes = gpu_runtimes / native_runtimes
    relative_no_record_runtimes = gpu_no_record_runtimes / native_no_record_runtimes

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_compartments, relative_runtimes, marker="o", label="with recording")
    ax.plot(n_compartments, relative_no_record_runtimes, marker="s", label="without recording")
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("added dendritic compartment count")
    ax.set_ylabel("relative runtime")
    activity = "active" if morphology == "chain" else "passive"
    ax.set_title(f"cm_default {activity} {morphology} compartment benchmark")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def print_population_run_header(n_neurons, sample_neuron):
    print("\n=== cm_default active population benchmark ===")
    print(f"setup: {n_neurons} neuron(s), two compartments per neuron, two AMPA_NMDA receptors per neuron")
    print(f"sample: neuron {sample_neuron}, active dendrite recordables")


def print_compartment_run_header(n_added_compartments, sample_compartment, morphology="chain"):
    activity = "active" if morphology == "chain" else "passive"
    print(f"\n=== cm_default {activity} {morphology} compartment-count benchmark ===")
    print(f"setup: one neuron, {n_added_compartments} added dendritic compartment(s), "
          f"{n_added_compartments + 1} total compartment(s)")
    if morphology == "chain":
        print("morphology: each dendrite is the child of the preceding compartment")
    else:
        print("morphology: every dendrite is a direct child of the soma")
    receptor = "AMPA_NMDA" if morphology == "chain" else "AMPA"
    print(f"receptors: one {receptor} receptor at soma, spike input connected to receptor port 0")
    recordables = "voltage and channel-state" if morphology == "chain" else "voltage"
    print(f"sample: compartment {sample_compartment}, {recordables} recordables")


def print_benchmark_results(title, size_label, results):
    print(f"\n=== {title} results ===")
    print(f"{size_label:>14}  {'sample':>8}  {'recording':>10}  "
          f"{'NEST [s]':>12}  {'NEST-GPU [s]':>14}  {'GPU/NEST':>10}")
    for result in results:
        if size_label == "neurons":
            size = result["n_neurons"]
            sample = result["sample_neuron"]
        else:
            size = result["n_added_compartments"]
            sample = result["sample_compartment"]
        for recording, native_key, gpu_key in (
                ("yes", "native_runtime", "gpu_runtime"),
                ("no", "native_no_record_runtime", "gpu_no_record_runtime")):
            native_runtime = result[native_key]
            gpu_runtime = result[gpu_key]
            ratio = gpu_runtime / native_runtime if native_runtime > 0 else float("inf")
            print(f"{size:14d}  {sample:8d}  {recording:>10}  {native_runtime:12.6f}  "
                  f"{gpu_runtime:14.6f}  {ratio:10.3f}")


class TestNESTGPUCompartmentalPopulationBenchmark:
    def test_cm_default_population_benchmark_against_native_nest(self, tmp_path, benchmark_target_path):
        rng = np.random.default_rng(BENCHMARK_RANDOM_SEED)
        benchmark_results = []
        for n_neurons in BENCHMARK_POPULATION_SIZES:
            sample_neuron = int(rng.integers(0, n_neurons))
            print_population_run_header(n_neurons, sample_neuron)
            print("running native NEST reference simulation")
            native = run_native_active_population_cm_default(n_neurons, sample_neuron)
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_population_cm_default(tmp_path, n_neurons, sample_neuron)
            compare_active_default_traces(native, gpu)
            print("running native NEST simulation without recording")
            native_no_record = run_native_active_population_cm_default(n_neurons, sample_neuron, record=False)
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_population_cm_default(
                tmp_path, n_neurons, sample_neuron, record=False)
            benchmark_results.append({
                "n_neurons": n_neurons,
                "sample_neuron": sample_neuron,
                "native_runtime": native["runtime"],
                "gpu_runtime": gpu["runtime"],
                "native_no_record_runtime": native_no_record["runtime"],
                "gpu_no_record_runtime": gpu_no_record["runtime"],
            })
            print(f"finished population run with recording: native={native['runtime']:.6f}s, "
                  f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / native['runtime']:.3f}")
            print(f"finished population run without recording: native={native_no_record['runtime']:.6f}s, "
                  f"gpu={gpu_no_record['runtime']:.6f}s, "
                  f"ratio={gpu_no_record['runtime'] / native_no_record['runtime']:.3f}")

        print_benchmark_results("cm_default active population benchmark", "neurons", benchmark_results)
        plot_population_benchmark(benchmark_results, benchmark_target_path)
        with open(os.path.join(benchmark_target_path, "gpu_compartmental_population_benchmark.json"), "w",
                  encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)

    def test_cm_default_compartment_benchmark_against_native_nest(self, tmp_path, benchmark_target_path):
        benchmark_results = []
        for n_added_compartments in BENCHMARK_COMPARTMENT_SIZES:
            sample_compartment = n_added_compartments
            print_compartment_run_header(n_added_compartments, sample_compartment)
            print("running native NEST reference simulation")
            native = run_native_active_compartment_cm_default(n_added_compartments, sample_compartment)
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_compartment_cm_default(tmp_path, n_added_compartments, sample_compartment)
            compare_active_compartment_default_traces(native, gpu, sample_compartment)
            print("running native NEST simulation without recording")
            native_no_record = run_native_active_compartment_cm_default(
                n_added_compartments, sample_compartment, record=False)
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, record=False)
            benchmark_results.append({
                "n_added_compartments": n_added_compartments,
                "n_compartments": n_added_compartments + 1,
                "sample_compartment": sample_compartment,
                "native_runtime": native["runtime"],
                "gpu_runtime": gpu["runtime"],
                "native_no_record_runtime": native_no_record["runtime"],
                "gpu_no_record_runtime": gpu_no_record["runtime"],
            })
            print(f"finished compartment run with recording: native={native['runtime']:.6f}s, "
                  f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / native['runtime']:.3f}")
            print(f"finished compartment run without recording: native={native_no_record['runtime']:.6f}s, "
                  f"gpu={gpu_no_record['runtime']:.6f}s, "
                  f"ratio={gpu_no_record['runtime'] / native_no_record['runtime']:.3f}")

        print_benchmark_results("cm_default active compartment-count benchmark", "added_comp", benchmark_results)
        plot_compartment_benchmark(benchmark_results, benchmark_target_path, morphology="chain")
        with open(os.path.join(benchmark_target_path, "gpu_compartmental_compartment_benchmark.json"), "w",
                  encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)

    def test_cm_default_star_compartment_benchmark_against_native_nest(self, tmp_path, benchmark_target_path):
        benchmark_results = []
        for n_added_compartments in BENCHMARK_COMPARTMENT_SIZES:
            sample_compartment = n_added_compartments
            print_compartment_run_header(n_added_compartments, sample_compartment, morphology="star")
            print("running native NEST reference simulation")
            native = run_native_active_compartment_cm_default(
                n_added_compartments, sample_compartment, morphology="star")
            print("running generated NEST-GPU simulation")
            gpu = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, morphology="star")
            compare_trace(
                native["active"], gpu["active"],
                f"v_comp{sample_compartment}", f"v_comp{sample_compartment}", atol=0.5)
            print("running native NEST simulation without recording")
            native_no_record = run_native_active_compartment_cm_default(
                n_added_compartments, sample_compartment, record=False, morphology="star")
            print("running generated NEST-GPU simulation without recording")
            gpu_no_record = run_gpu_active_compartment_cm_default(
                tmp_path, n_added_compartments, sample_compartment, record=False, morphology="star")
            benchmark_results.append({
                "n_added_compartments": n_added_compartments,
                "n_compartments": n_added_compartments + 1,
                "sample_compartment": sample_compartment,
                "morphology": "star",
                "native_runtime": native["runtime"],
                "gpu_runtime": gpu["runtime"],
                "native_no_record_runtime": native_no_record["runtime"],
                "gpu_no_record_runtime": gpu_no_record["runtime"],
            })
            print(f"finished star compartment run with recording: native={native['runtime']:.6f}s, "
                  f"gpu={gpu['runtime']:.6f}s, ratio={gpu['runtime'] / native['runtime']:.3f}")
            print(f"finished star compartment run without recording: "
                  f"native={native_no_record['runtime']:.6f}s, gpu={gpu_no_record['runtime']:.6f}s, "
                  f"ratio={gpu_no_record['runtime'] / native_no_record['runtime']:.3f}")

        print_benchmark_results(
            "cm_default passive star compartment-count benchmark", "added_comp", benchmark_results)
        plot_compartment_benchmark(benchmark_results, benchmark_target_path, morphology="star")
        with open(os.path.join(
                benchmark_target_path, "gpu_compartmental_star_compartment_benchmark.json"), "w",
                encoding="utf-8") as output_file:
            json.dump(benchmark_results, output_file, indent=2)
