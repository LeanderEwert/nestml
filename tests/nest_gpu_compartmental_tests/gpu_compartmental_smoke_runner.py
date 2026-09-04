# -*- coding: utf-8 -*-
#
# gpu_compartmental_smoke_runner.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

import json
import sys


RECEPTOR_MODEL_NAME = "cm_ampa_only_nestml"
CHANNEL_MODEL_NAME = "cm_channel_only_nestml"
CONTINUOUS_MODEL_NAME = "cm_continuous_only_nestml"
CONCENTRATION_MODEL_NAME = "cm_concentration_only_nestml"
DEPENDENCY_MODEL_NAME = "cm_dependency_nestml"
DEFAULT_MODEL_NAME = "cm_default_nestml"
DEFAULT_DT = 0.001
DEFAULT_SIM_TIME = 160.0

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


def run_receptor_only_simulation_smoke():
    import nestgpu as ngpu

    neuron = ngpu.Create(RECEPTOR_MODEL_NAME, 1)

    ngpu.SetStatus(neuron, {
        "compartments": [{"parent_idx": -1, "params": {}}],
        "receptors": [{"comp_idx": 0, "receptor_type": "AMPA"}],
    })
    record = ngpu.CreateRecord("", ["v_comp0", "g_AMPA0", "i_tot_AMPA0"], [neuron[0], neuron[0], neuron[0]], [0, 0, 0])

    spike_generator = ngpu.Create("spike_generator")
    ngpu.SetStatus(spike_generator, {"spike_times": list(range(1, 5))})

    conn_dict = {"rule": "one_to_one"}
    syn_dict = {"weight": 1.0, "delay": 2.0, "receptor": 0}

    ngpu.Connect(spike_generator, neuron, conn_dict, syn_dict)
    ngpu.Simulate(6.0)
    recorded_data = ngpu.GetRecordData(record)
    if len(recorded_data) == 0 or len(recorded_data[0]) != 4:
        raise AssertionError("v_comp0/g_AMPA0/i_tot_AMPA0 record data was not produced")


def run_channel_only_simulation_smoke():
    import nestgpu as ngpu

    neuron = ngpu.Create(CHANNEL_MODEL_NAME, 1)

    ngpu.SetStatus(neuron, {
        "compartments": [{
            "parent_idx": -1,
            "params": {
                "gbar_leak": 0.1,
                "e_leak": -70.0,
            },
        }],
    })

    ngpu.Simulate(2.0)


def run_continuous_only_simulation_smoke():
    import nestgpu as ngpu

    neuron = ngpu.Create(CONTINUOUS_MODEL_NAME, 1)

    ngpu.SetStatus(neuron, {
        "compartments": [{"parent_idx": -1, "params": {}}],
        "continuous_inputs": [{"comp_idx": 0, "continuous_input_type": "I_ext"}],
    })

    ngpu.Simulate(2.0)


def run_concentration_only_simulation_smoke():
    import nestgpu as ngpu

    neuron = ngpu.Create(CONCENTRATION_MODEL_NAME, 1)

    ngpu.SetStatus(neuron, {
        "compartments": [{
            "parent_idx": -1,
            "params": {
                "c_Ca": 0.0002,
                "inf_Ca": 0.0001,
                "tau_Ca": 10.0,
            },
        }],
    })

    ngpu.Simulate(2.0)


def run_dependency_simulation_smoke():
    import nestgpu as ngpu

    neuron = ngpu.Create(DEPENDENCY_MODEL_NAME, 1)

    ngpu.SetStatus(neuron, {
        "compartments": [{
            "parent_idx": -1,
            "params": {
                "c_Ca": 0.0002,
                "gbar_leak": 0.1,
                "e_leak": -70.0,
                "inf_Ca": 0.0001,
                "tau_Ca": 10.0,
                "gamma_Ca": 0.001,
            },
        }],
    })

    ngpu.Simulate(2.0)


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

    passive_recordables = [
        "v_comp0",
        "v_comp1",
        "m_Na0",
        "h_Na0",
        "n_K0",
        "g_AN_AMPA1",
        "g_AN_NMDA1",
    ]
    active_recordables = [
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
    pas_record = ngpu.CreateRecord("", passive_recordables, [cm_pas[0]] * len(passive_recordables), [0] * len(passive_recordables))
    act_record = ngpu.CreateRecord("", active_recordables, [cm_act[0]] * len(active_recordables), [0] * len(active_recordables))

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
    _assert_record_data(pas_data, passive_recordables)
    _assert_record_data(act_data, active_recordables)
    return {
        "passive": _record_data_to_dict(pas_data, passive_recordables),
        "active": _record_data_to_dict(act_data, active_recordables),
    }


def run_default_simulation_smoke():
    run_default_simulation()


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "usage: gpu_compartmental_smoke_runner.py "
            "[receptor|channel|continuous|concentration|dependency|default|default-json] [output.json]")

    if sys.argv[1] == "receptor":
        run_receptor_only_simulation_smoke()
    elif sys.argv[1] == "channel":
        run_channel_only_simulation_smoke()
    elif sys.argv[1] == "continuous":
        run_continuous_only_simulation_smoke()
    elif sys.argv[1] == "concentration":
        run_concentration_only_simulation_smoke()
    elif sys.argv[1] == "dependency":
        run_dependency_simulation_smoke()
    elif sys.argv[1] == "default":
        run_default_simulation_smoke()
    elif sys.argv[1] == "default-json":
        if len(sys.argv) != 3:
            raise SystemExit("default-json requires an output path")
        with open(sys.argv[2], "w", encoding="utf-8") as output_file:
            json.dump(run_default_simulation(), output_file)
    else:
        raise SystemExit("unknown smoke test: " + sys.argv[1])
