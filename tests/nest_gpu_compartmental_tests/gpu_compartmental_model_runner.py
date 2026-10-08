# -*- coding: utf-8 -*-
#
# gpu_compartmental_model_runner.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

import ctypes
import json
import sys
import time


DEFAULT_MODEL_NAME = "cm_default_nestml"
DEFAULT_DT = 0.1
DEFAULT_SIM_TIME = 1000.0
COMPARTMENT_BENCHMARK_SPIKE_TIMES = [10.0, 13.0, 16.0]

SOMA_PARAMS = {
    "C_m": 89.245535,
    "g_C": 0.0,
    "g_L": 8.924572508,
    "e_L": -75.0,
    "gbar_Na": 4608.698576715,
    "e_Na": 60.0,
    "gbar_K": 956.112772900,
    "e_K": -90.0,
}

SOMA_PARAMS_PASSIVE = {
    "C_m": SOMA_PARAMS["C_m"],
    "g_C": SOMA_PARAMS["g_C"],
    "g_L": SOMA_PARAMS["g_L"],
    "e_L": SOMA_PARAMS["e_L"],
}

DEND_PARAMS_PASSIVE = {
    "C_m": 1.929929,
    "g_C": 1.255439494,
    "g_L": 0.192992878,
    "e_L": -75.0,
}

DEND_PARAMS_ACTIVE = {
    "C_m": 1.929929,
    "g_C": 1.255439494,
    "g_L": 0.192992878,
    "e_L": -75.0,
    "gbar_Na": 17.203212493,
    "e_Na": 60.0,
    "gbar_K": 11.887347450,
    "e_K": -90.0,
}

PASSIVE_RECORDABLES = [
    "v_comp0",
    "v_comp1",
    "m_Na0",
    "h_Na0",
    "n_K0",
    "g_AN_AMPA1",
    "g_AN_NMDA1",
]

ACTIVE_RECORDABLES = [
    "v_comp0",
    "v_comp1",
    "m_Na0",
    "h_Na0",
    "n_K0",
    "m_Na1",
    "h_Na1",
    "n_K1",
    "g_AN_AMPA1",
    "g_AN_NMDA1",
]


def benchmark_neuron_configuration(n_added_compartments, morphology="chain", dynamics="active"):
    """Shared CPU/GPU sweep configuration: uniform dynamics and somatic AMPA."""
    if n_added_compartments < 0 or morphology not in ("chain", "star") or dynamics not in ("active", "passive"):
        raise ValueError("Invalid benchmark morphology or dynamics")
    soma = dict(SOMA_PARAMS if dynamics == "active" else SOMA_PARAMS_PASSIVE)
    dendrite = dict(DEND_PARAMS_ACTIVE if dynamics == "active" else DEND_PARAMS_PASSIVE)
    for params in (soma, dendrite):
        params["v_comp"] = -75.0
        if dynamics == "passive":
            params.update({"gbar_Na": 0.0, "gbar_K": 0.0})
    return {
        "V_th": -50.0,
        "compartments": [{"parent_idx": -1, "params": soma}] + [
            {"parent_idx": index - 1 if morphology == "chain" else 0, "params": dict(dendrite)}
            for index in range(1, n_added_compartments + 1)],
        "receptors": [{"comp_idx": 0, "receptor_type": "AMPA",
                       "params": {"e_AMPA": 0.0, "tau_r_AMPA": 0.2, "tau_d_AMPA": 3.0}}],
    }


def _synchronize(ngpu):
    library = ctypes.CDLL(ngpu.lib_path)
    synchronize = library.cudaDeviceSynchronize
    synchronize.restype = ctypes.c_int
    synchronize.argtypes = []
    if synchronize():
        raise RuntimeError("CUDA synchronization failed")


def _check_benchmark_mechanisms(ngpu, neuron, n_compartments, dynamics):
    # Runtime names enumerate instantiated mechanisms, including unrecorded ones.
    names = [name.decode() if isinstance(name, bytes) else name for name in ngpu.GetScalVarNames(neuron)]
    expected = n_compartments if dynamics == "active" else 0
    for gate in ("m_Na", "h_Na", "n_K"):
        assert sum(name.startswith(gate) for name in names) == expected, (gate, dynamics)
    assert not any("NMDA" in name for name in names)


def _configure_default_neuron(ngpu, neuron, dend_params):
    ngpu.SetStatus(neuron, {
        "V_th": -50.0,
        "compartments": [
            {"parent_idx": -1, "params": SOMA_PARAMS},
            {"parent_idx": 0, "params": dend_params},
        ],
        "receptors": [
            {"comp_idx": 0, "receptor_type": "AMPA_NMDA"},
            {"comp_idx": 1, "receptor_type": "AMPA_NMDA"},
        ],
    })


def _assert_record_data(recorded_data, recordables):
    if len(recorded_data) == 0:
        raise AssertionError("record data was not produced")
    expected_columns = len(recordables) + 1
    if len(recorded_data[0]) != expected_columns:
        raise AssertionError(f"Expected {expected_columns} record columns, got {len(recorded_data[0])}")


def _record_data_to_dict(recorded_data, recordables):
    result = {"times": [row[0] for row in recorded_data]}
    for i, recordable in enumerate(recordables, start=1):
        result[recordable] = [row[i] for row in recorded_data]
    return result


def run_default_simulation():
    import nestgpu as ngpu

    ngpu.SetTimeResolution(DEFAULT_DT)

    cm_pas = ngpu.Create(DEFAULT_MODEL_NAME, 1)
    cm_act = ngpu.Create(DEFAULT_MODEL_NAME, 1)

    _configure_default_neuron(ngpu, cm_pas, DEND_PARAMS_PASSIVE)
    _configure_default_neuron(ngpu, cm_act, DEND_PARAMS_ACTIVE)

    pas_record = ngpu.CreateRecord("", PASSIVE_RECORDABLES, [cm_pas[0]] * len(PASSIVE_RECORDABLES),
                                   [0] * len(PASSIVE_RECORDABLES))
    act_record = ngpu.CreateRecord("", ACTIVE_RECORDABLES, [cm_act[0]] * len(ACTIVE_RECORDABLES),
                                   [0] * len(ACTIVE_RECORDABLES))

    sg_soma = ngpu.Create("spike_generator")
    sg_dend = ngpu.Create("spike_generator")
    ngpu.SetStatus(sg_soma, {"spike_times": [10.0, 13.0, 16.0]})
    ngpu.SetStatus(sg_dend, {"spike_times": [70.0, 73.0, 76.0]})

    conn_dict = {"rule": "one_to_one"}
    ngpu.Connect(sg_soma, cm_pas, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})
    ngpu.Connect(sg_dend, cm_pas, conn_dict, {"weight": 2.0, "delay": 0.5, "receptor": 1})
    ngpu.Connect(sg_soma, cm_act, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})
    ngpu.Connect(sg_dend, cm_act, conn_dict, {"weight": 2.0, "delay": 0.5, "receptor": 1})

    ngpu.Simulate(DEFAULT_SIM_TIME)

    pas_data = ngpu.GetRecordData(pas_record)
    act_data = ngpu.GetRecordData(act_record)
    _assert_record_data(pas_data, PASSIVE_RECORDABLES)
    _assert_record_data(act_data, ACTIVE_RECORDABLES)
    return {
        "passive": _record_data_to_dict(pas_data, PASSIVE_RECORDABLES),
        "active": _record_data_to_dict(act_data, ACTIVE_RECORDABLES),
    }


def run_active_population_simulation(n_neurons, sample_neuron, record=True, dynamics="active"):
    import nestgpu as ngpu

    if sample_neuron < 0 or sample_neuron >= n_neurons:
        raise ValueError("sample_neuron must be inside the created population")

    ngpu.SetTimeResolution(DEFAULT_DT)
    _synchronize(ngpu)
    t_start = time.perf_counter()

    neurons = ngpu.Create(DEFAULT_MODEL_NAME, n_neurons)
    for i_neuron in range(n_neurons):
        ngpu.SetStatus(neurons[i_neuron:i_neuron + 1], benchmark_neuron_configuration(1, dynamics=dynamics))

    recordables = ["v_comp0", "v_comp1"]
    if record:
        recorder = ngpu.CreateRecord("", recordables,
                                     [neurons[sample_neuron]] * len(recordables), [0] * len(recordables))

    sg_soma = ngpu.Create("spike_generator", n_neurons)
    ngpu.SetStatus(sg_soma, {"spike_times": COMPARTMENT_BENCHMARK_SPIKE_TIMES})

    conn_dict = {"rule": "one_to_one"}
    ngpu.Connect(sg_soma, neurons, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})

    ngpu.Simulate(DEFAULT_SIM_TIME)

    _synchronize(ngpu)
    runtime = time.perf_counter() - t_start
    _check_benchmark_mechanisms(ngpu, neurons[0], 2, dynamics)
    recorded_data = ngpu.GetRecordData(recorder) if record else None

    result = {
        "n_neurons": n_neurons,
        "sample_neuron": sample_neuron,
        "recording_enabled": record,
        "dynamics": dynamics,
        "runtime": runtime,
    }
    if record:
        _assert_record_data(recorded_data, recordables)
        result["traces"] = _record_data_to_dict(recorded_data, recordables)
    return result


def run_active_compartment_simulation(
        n_added_compartments, sample_compartment, record=True, morphology="chain", dynamics=None):
    import nestgpu as ngpu

    n_compartments = n_added_compartments + 1
    if sample_compartment < 0 or sample_compartment >= n_compartments:
        raise ValueError("sample_compartment must be inside the created morphology")

    dynamics = dynamics or ("passive" if morphology == "star" else "active")
    recordables = [f"v_comp{sample_compartment}"]

    ngpu.SetTimeResolution(DEFAULT_DT)
    _synchronize(ngpu)
    t_start = time.perf_counter()

    neuron = ngpu.Create(DEFAULT_MODEL_NAME, 1)
    ngpu.SetStatus(neuron, benchmark_neuron_configuration(n_added_compartments, morphology, dynamics))

    if record:
        recorder = ngpu.CreateRecord("", recordables, [neuron[0]] * len(recordables), [0] * len(recordables))

    spike_generator = ngpu.Create("spike_generator")
    ngpu.SetStatus(spike_generator, {"spike_times": COMPARTMENT_BENCHMARK_SPIKE_TIMES})

    conn_dict = {"rule": "one_to_one"}
    ngpu.Connect(spike_generator, neuron, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})

    ngpu.Simulate(DEFAULT_SIM_TIME)

    _synchronize(ngpu)
    runtime = time.perf_counter() - t_start
    _check_benchmark_mechanisms(ngpu, neuron[0], n_compartments, dynamics)
    recorded_data = ngpu.GetRecordData(recorder) if record else None

    result = {
        "n_added_compartments": n_added_compartments,
        "n_compartments": n_compartments,
        "sample_compartment": sample_compartment,
        "morphology": morphology,
        "dynamics": dynamics,
        "recording_enabled": record,
        "runtime": runtime,
    }
    if record:
        _assert_record_data(recorded_data, recordables)
        result["traces"] = _record_data_to_dict(recorded_data, recordables)
    return result


if __name__ == "__main__":
    # Retain the existing commands and extend them symmetrically to both dynamics.
    modes = {}
    for dynamics in ("active", "passive"):
        for record in (True, False):
            suffix = "json" if record else "no-record-json"
            modes[f"{dynamics}-population-{suffix}"] = ("population", dynamics, record)
            modes[f"{dynamics}-compartment-{suffix}"] = ("chain", dynamics, record)
            modes[f"{dynamics}-star-compartment-{suffix}"] = ("star", dynamics, record)
    if len(sys.argv) == 3 and sys.argv[1] == "default-json":
        result = run_default_simulation()
    elif len(sys.argv) == 5 and sys.argv[1] in modes:
        morphology, dynamics, record = modes[sys.argv[1]]
        size, sample = int(sys.argv[3]), int(sys.argv[4])
        if morphology == "population":
            result = run_active_population_simulation(size, sample, record=record, dynamics=dynamics)
        else:
            result = run_active_compartment_simulation(
                size, sample, record=record, morphology=morphology, dynamics=dynamics)
    else:
        raise SystemExit("usage: gpu_compartmental_model_runner.py default-json output.json OR "
                         "<active|passive>-<population|compartment|star-compartment>-"
                         "<json|no-record-json> output.json size sample_index")
    with open(sys.argv[2], "w", encoding="utf-8") as output_file:
        json.dump(result, output_file)
