# -*- coding: utf-8 -*-
#
# gpu_compartmental_solver_runner.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) either version.

"""End-to-end driver for the compartmental tree solvers (plain Hines and R-EDD).

The runner only talks to NEST-GPU. The neuron morphology is supplied as a status
dict by the test through :func:`build_neuron_configuration`; this file does not
own any model description, so the same morphology can be exercised under either
solver. A mostly-passive morphology is initialised off equilibrium; the
resulting passive decay produces voltage fluctuations that propagate through the
dendritic tree and are recorded from several compartments.

Run a single configuration::

    python gpu_compartmental_solver_runner.py <config.json> <output.json>
"""

import ctypes
import json
import sys
import time

GPU_MODEL_NAME = "cm_default_nestml"
DT = 0.1
SIM_TIME = 1000.0

# Resting potential (also e_L). The soma is initialised significantly above it,
# so the passive leak drives a decay that propagates through the dendritic tree.
REST_VOLTAGE = -75.0
OFF_EQUILIBRIUM_VOLTAGE = -30.0

# Mostly passive: no active sodium/potassium channels, only a leak. The soma is
# initialised off equilibrium; the resulting passive decay drives the voltage
# fluctuations that exercise the tree solver across the dendrites.
SOMA_PARAMS = {
    "C_m": 89.245535,
    "g_C": 0.0,
    "g_L": 8.924572508,
    "e_L": REST_VOLTAGE,
    "v_comp": OFF_EQUILIBRIUM_VOLTAGE,
}

DEND_PARAMS_PASSIVE = {
    "C_m": 1.929929,
    "g_C": 1.255439494,
    "g_L": 0.192992878,
    "e_L": REST_VOLTAGE,
    "v_comp": REST_VOLTAGE,
}


def build_neuron_configuration(morphology, n_added_compartments, initial_voltage=OFF_EQUILIBRIUM_VOLTAGE):
    """Return a NEST-GPU status dict for one neuron with a dendritic tree.

    ``n_added_compartments`` dendrites are appended to the soma (compartment 0).
    ``chain`` links each new compartment to the preceding one (a long linear
    tree, ideal for R-EDD's chain decomposition), ``star`` makes every new
    compartment a direct child of the soma, and ``binary`` builds a balanced
    binary tree. Setting the soma's ``v_comp`` to ``initial_voltage`` places the
    cell off equilibrium; the passive decay that follows drives the voltage
    fluctuations that exercise the tree solver across the dendrites.
    """
    if morphology not in ("chain", "star", "binary"):
        raise ValueError(f"Unknown compartment morphology: {morphology}")
    if n_added_compartments < 1:
        raise ValueError("n_added_compartments must be >= 1")

    soma = dict(SOMA_PARAMS)
    soma["v_comp"] = initial_voltage
    dend = dict(DEND_PARAMS_PASSIVE)
    compartments = [{"parent_idx": -1, "params": soma}]
    for index in range(1, 1 + n_added_compartments):
        if morphology == "chain":
            parent = index - 1
        elif morphology == "star":
            parent = 0
        else:  # binary: index 1 is the soma's child, then split
            parent = (index - 1) // 2
        compartments.append({"parent_idx": parent, "params": dend})

    return {
        "compartments": compartments,
    }


def _synchronize(ngpu):
    library = ctypes.CDLL(ngpu.lib_path)
    synchronize = library.cudaDeviceSynchronize
    synchronize.restype = ctypes.c_int
    synchronize.argtypes = []
    if synchronize():
        raise RuntimeError("CUDA synchronization failed")


def _record_data_to_dict(recorded_data, recordables):
    result = {"times": [row[0] for row in recorded_data]}
    for i, recordable in enumerate(recordables, start=1):
        result[recordable] = [row[i] for row in recorded_data]
    return result


def run_solver_simulation(config, sample_compartments, dt=DT, sim_time=SIM_TIME):
    """Create one neuron and return the recorded voltages of its tree.

    The neuron is initialised off equilibrium (see
    :func:`build_neuron_configuration`); the resulting passive decay propagates
    through the dendritic tree. The returned dict maps each sampled compartment
    to ``{"times": [...], "<name>": [...]}``, aligned by time, ready for
    ``numpy.interp``-style comparison between solvers.
    """
    import nestgpu as ngpu

    recordables = [f"v_comp{index}" for index in sample_compartments]

    _synchronize(ngpu)
    t_start = time.perf_counter()

    neuron = ngpu.Create(GPU_MODEL_NAME, 1)
    ngpu.SetStatus(neuron, config)

    recorder = ngpu.CreateRecord("", recordables,
                                 [neuron[0]] * len(recordables), [0] * len(recordables))

    ngpu.Simulate(sim_time)

    _synchronize(ngpu)
    runtime = time.perf_counter() - t_start
    recorded_data = ngpu.GetRecordData(recorder)

    result = {
        "morphology": config.get("morphology", None),
        "n_added_compartments": len(config["compartments"]) - 1,
        "sample_compartments": list(sample_compartments),
        "dynamics": "passive",
        "recording_enabled": True,
        "runtime": runtime,
    }
    result["traces"] = _record_data_to_dict(recorded_data, recordables)
    return result


def run_from_params(params):
    """Run one scenario from a JSON-serializable ``params`` dict and return the result.

    ``params`` carries the full configuration (see :func:`build_neuron_configuration`)
    plus the scenario fields ``sample_compartments``, ``dt`` and ``sim_time``. A single
    scenario per process is required: NEST-GPU calibrates its neuron graph on the first
    ``Create`` and refuses to create new nodes afterwards, so each scenario must run in
    its own process.
    """
    config = dict(params.get("config", {}))
    config["morphology"] = config.get("morphology")
    return run_solver_simulation(
        config,
        sample_compartments=params.get("sample_compartments", [0, len(config["compartments"]) - 1]),
        dt=params.get("dt", DT),
        sim_time=params.get("sim_time", SIM_TIME),
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: gpu_compartmental_solver_runner.py <params.json> <output.json>")
    with open(sys.argv[1], encoding="utf-8") as input_file:
        params = json.load(input_file)
    result = run_from_params(params)
    with open(sys.argv[2], "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
