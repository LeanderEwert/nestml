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

import json
import sys
import time


DEFAULT_MODEL_NAME = "cm_default_nestml"
DEFAULT_DT = 0.001
DEFAULT_SIM_TIME = 160.0
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


def _active_recordables_for_compartment(compartment):
    return [
        f"v_comp{compartment}",
        f"m_Na{compartment}",
        f"h_Na{compartment}",
        f"n_K{compartment}",
    ]


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


def _configure_active_compartment_chain(ngpu, neuron, n_added_compartments):
    n_compartments = n_added_compartments + 1
    compartments = [{"parent_idx": -1, "params": SOMA_PARAMS}]
    for compartment in range(1, n_compartments):
        compartments.append({"parent_idx": compartment - 1, "params": DEND_PARAMS_ACTIVE})

    ngpu.SetStatus(neuron, {
        "V_th": -50.0,
        "compartments": compartments,
        "receptors": [
            {"comp_idx": 0, "receptor_type": "AMPA_NMDA"},
        ],
    })


def _configure_passive_compartment_star(ngpu, neuron, n_added_compartments):
    compartments = [{"parent_idx": -1, "params": SOMA_PARAMS_PASSIVE}]
    compartments.extend(
        {"parent_idx": 0, "params": DEND_PARAMS_PASSIVE}
        for _ in range(n_added_compartments)
    )

    ngpu.SetStatus(neuron, {
        "V_th": -50.0,
        "compartments": compartments,
        "receptors": [
            {"comp_idx": 0, "receptor_type": "AMPA"},
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


def run_active_population_simulation(n_neurons, sample_neuron, record=True):
    import nestgpu as ngpu

    if sample_neuron < 0 or sample_neuron >= n_neurons:
        raise ValueError("sample_neuron must be inside the created population")

    t_start = time.perf_counter()
    ngpu.SetTimeResolution(DEFAULT_DT)

    neurons = ngpu.Create(DEFAULT_MODEL_NAME, n_neurons)
    for i_neuron in range(n_neurons):
        _configure_default_neuron(ngpu, neurons[i_neuron:i_neuron + 1], DEND_PARAMS_ACTIVE)

    if record:
        recorder = ngpu.CreateRecord("", ACTIVE_RECORDABLES,
                                     [neurons[sample_neuron]] * len(ACTIVE_RECORDABLES),
                                     [0] * len(ACTIVE_RECORDABLES))

    sg_soma = ngpu.Create("spike_generator", n_neurons)
    sg_dend = ngpu.Create("spike_generator", n_neurons)
    ngpu.SetStatus(sg_soma, {"spike_times": [10.0, 13.0, 16.0]})
    ngpu.SetStatus(sg_dend, {"spike_times": [70.0, 73.0, 76.0]})

    conn_dict = {"rule": "one_to_one"}
    ngpu.Connect(sg_soma, neurons, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})
    ngpu.Connect(sg_dend, neurons, conn_dict, {"weight": 2.0, "delay": 0.5, "receptor": 1})

    ngpu.Simulate(DEFAULT_SIM_TIME)

    recorded_data = ngpu.GetRecordData(recorder) if record else None
    runtime = time.perf_counter() - t_start

    result = {
        "n_neurons": n_neurons,
        "sample_neuron": sample_neuron,
        "recording_enabled": record,
        "runtime": runtime,
    }
    if record:
        _assert_record_data(recorded_data, ACTIVE_RECORDABLES)
        result["active"] = _record_data_to_dict(recorded_data, ACTIVE_RECORDABLES)
    return result


def run_active_compartment_simulation(n_added_compartments, sample_compartment, record=True, morphology="chain"):
    import nestgpu as ngpu

    n_compartments = n_added_compartments + 1
    if sample_compartment < 0 or sample_compartment >= n_compartments:
        raise ValueError("sample_compartment must be inside the created morphology")

    recordables = ([f"v_comp{sample_compartment}"] if morphology == "star"
                   else _active_recordables_for_compartment(sample_compartment))

    t_start = time.perf_counter()
    ngpu.SetTimeResolution(DEFAULT_DT)

    neuron = ngpu.Create(DEFAULT_MODEL_NAME, 1)
    if morphology == "chain":
        _configure_active_compartment_chain(ngpu, neuron, n_added_compartments)
    elif morphology == "star":
        _configure_passive_compartment_star(ngpu, neuron, n_added_compartments)
    else:
        raise ValueError(f"Unknown compartment morphology: {morphology}")

    if record:
        recorder = ngpu.CreateRecord("", recordables, [neuron[0]] * len(recordables), [0] * len(recordables))

    spike_generator = ngpu.Create("spike_generator")
    ngpu.SetStatus(spike_generator, {"spike_times": COMPARTMENT_BENCHMARK_SPIKE_TIMES})

    conn_dict = {"rule": "one_to_one"}
    ngpu.Connect(spike_generator, neuron, conn_dict, {"weight": 5.0, "delay": 0.5, "receptor": 0})

    ngpu.Simulate(DEFAULT_SIM_TIME)

    recorded_data = ngpu.GetRecordData(recorder) if record else None
    runtime = time.perf_counter() - t_start

    result = {
        "n_added_compartments": n_added_compartments,
        "n_compartments": n_compartments,
        "sample_compartment": sample_compartment,
        "morphology": morphology,
        "recording_enabled": record,
        "runtime": runtime,
    }
    if record:
        _assert_record_data(recorded_data, recordables)
        result["active"] = _record_data_to_dict(recorded_data, recordables)
    return result


if __name__ == "__main__":
    if len(sys.argv) not in (3, 5):
        raise SystemExit(
            "usage: gpu_compartmental_model_runner.py "
            "[default-json|active-population-json|active-population-no-record-json|"
            "active-compartment-json|active-compartment-no-record-json|"
            "passive-star-compartment-json|passive-star-compartment-no-record-json] "
            "output.json [size sample_index]")

    if sys.argv[1] == "default-json":
        if len(sys.argv) != 3:
            raise SystemExit("default-json requires an output path")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_default_simulation(), output_file)
    elif sys.argv[1] == "active-population-json":
        if len(sys.argv) != 5:
            raise SystemExit("active-population-json requires output path, n_neurons, and sample_neuron")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_population_simulation(int(sys.argv[3]), int(sys.argv[4])), output_file)
    elif sys.argv[1] == "active-population-no-record-json":
        if len(sys.argv) != 5:
            raise SystemExit(
                "active-population-no-record-json requires output path, n_neurons, and sample_neuron")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_population_simulation(int(sys.argv[3]), int(sys.argv[4]), record=False),
                      output_file)
    elif sys.argv[1] == "active-compartment-json":
        if len(sys.argv) != 5:
            raise SystemExit(
                "active-compartment-json requires output path, n_added_compartments, and sample_compartment")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_compartment_simulation(int(sys.argv[3]), int(sys.argv[4])), output_file)
    elif sys.argv[1] == "active-compartment-no-record-json":
        if len(sys.argv) != 5:
            raise SystemExit(
                "active-compartment-no-record-json requires output path, n_added_compartments, and "
                "sample_compartment")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_compartment_simulation(int(sys.argv[3]), int(sys.argv[4]), record=False),
                      output_file)
    elif sys.argv[1] == "passive-star-compartment-json":
        if len(sys.argv) != 5:
            raise SystemExit(
                "passive-star-compartment-json requires output path, n_added_compartments, and "
                "sample_compartment")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_compartment_simulation(
                int(sys.argv[3]), int(sys.argv[4]), morphology="star"), output_file)
    elif sys.argv[1] == "passive-star-compartment-no-record-json":
        if len(sys.argv) != 5:
            raise SystemExit(
                "passive-star-compartment-no-record-json requires output path, n_added_compartments, and "
                "sample_compartment")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_active_compartment_simulation(
                int(sys.argv[3]), int(sys.argv[4]), record=False, morphology="star"), output_file)
    else:
        raise SystemExit("unknown model runner command: " + sys.argv[1])
