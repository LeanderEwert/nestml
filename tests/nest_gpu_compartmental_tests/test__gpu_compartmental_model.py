# -*- coding: utf-8 -*-
#
# test__gpu_compartmental_model.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

import json
import os
import shutil
import subprocess
import sys

import nest
import numpy as np

from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target


TEST_PLOTS = True
PLOT_IMPORT_ERROR = None
try:
    import matplotlib.pyplot as plt
except BaseException as e:
    TEST_PLOTS = False
    PLOT_IMPORT_ERROR = e

DT = 0.001
SIM_TIME = 160.0
GPU_MODEL_NAME = "cm_default_nestml"

SOMA_PARAMS = {
    "C_m": 89.245535,
    "g_C": 0.0,
    "g_L": 8.924572508,
    "e_L": -75.0,
    "gbar_Na": 4608.698576715,
    "e_Na": 60.0,
    "gbar_K": 956.112772900,
    "e_K": -90.0,
    "v_comp": -75.0,
}

DEND_PARAMS_PASSIVE = {
    "C_m": 1.929929,
    "g_C": 1.255439494,
    "g_L": 0.192992878,
    "e_L": -75.0,
    "v_comp": -75.0,
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
    "v_comp": -75.0,
}

PASSIVE_GPU_RECORDABLES = [
    "v_comp0",
    "v_comp1",
    "m_Na0",
    "h_Na0",
    "n_K0",
    "g_AN_AMPA1",
    "g_AN_NMDA1",
]

ACTIVE_GPU_RECORDABLES = [
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

NEST_RECORDABLES = [
    "v_comp0",
    "v_comp1",
    "m_Na_0",
    "h_Na_0",
    "n_K_0",
    "m_Na_1",
    "h_Na_1",
    "n_K_1",
    "g_r_AN_AMPA_1",
    "g_d_AN_AMPA_1",
    "g_r_AN_NMDA_1",
    "g_d_AN_NMDA_1",
]


def generate_gpu_default_model(target_path):
    tests_path = os.path.realpath(os.path.dirname(__file__))
    input_path = os.path.join(tests_path, "resources", "cm_default.nestml")
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)

    generate_nest_gpu_compartmental_target(
        input_path=input_path,
        target_path=target_path,
        module_name="cm_default_gpu_module",
        suffix="_nestml",
        logging_level="ERROR",
        dev=True,
        codegen_opts={
            "register_neuron_model": True,
            "skip_build": False,
        },
    )


def configure_native_neuron(neuron, dend_params):
    neuron.compartments = [
        {"parent_idx": -1, "params": SOMA_PARAMS},
        {"parent_idx": 0, "params": dend_params},
    ]
    neuron.V_th = -50.0
    neuron.receptors = [
        {"comp_idx": 0, "receptor_type": "AMPA_NMDA"},
        {"comp_idx": 1, "receptor_type": "AMPA_NMDA"},
    ]


def run_native_cm_default():
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT})

    cm_pas = nest.Create("cm_default")
    cm_act = nest.Create("cm_default")
    configure_native_neuron(cm_pas, DEND_PARAMS_PASSIVE)
    configure_native_neuron(cm_act, DEND_PARAMS_ACTIVE)

    sg_soma = nest.Create("spike_generator", 1, {"spike_times": [10.0, 13.0, 16.0]})
    sg_dend = nest.Create("spike_generator", 1, {"spike_times": [70.0, 73.0, 76.0]})

    nest.Connect(sg_soma, cm_pas, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})
    nest.Connect(sg_dend, cm_pas, syn_spec={
        "synapse_model": "static_synapse", "weight": 2.0, "delay": 0.5, "receptor_type": 1})
    nest.Connect(sg_soma, cm_act, syn_spec={
        "synapse_model": "static_synapse", "weight": 5.0, "delay": 0.5, "receptor_type": 0})
    nest.Connect(sg_dend, cm_act, syn_spec={
        "synapse_model": "static_synapse", "weight": 2.0, "delay": 0.5, "receptor_type": 1})

    mm_pas = nest.Create("multimeter", 1, {"record_from": NEST_RECORDABLES, "interval": DT})
    mm_act = nest.Create("multimeter", 1, {"record_from": NEST_RECORDABLES, "interval": DT})
    nest.Connect(mm_pas, cm_pas)
    nest.Connect(mm_act, cm_act)

    nest.Simulate(SIM_TIME)

    return {
        "passive": nest.GetStatus(mm_pas, "events")[0],
        "active": nest.GetStatus(mm_act, "events")[0],
    }


def run_gpu_cm_default(tmp_path):
    tests_path = os.path.realpath(os.path.dirname(__file__))
    runner = os.path.join(tests_path, "gpu_compartmental_model_runner.py")
    output_path = tmp_path / "cm_default_gpu_results.json"
    subprocess.check_call([sys.executable, runner, "default-json", str(output_path)])
    with open(output_path, encoding="utf-8") as input_file:
        return json.load(input_file)


def compare_trace(native_events, gpu_events, native_name, gpu_name, atol):
    native_times = np.asarray(native_events["times"], dtype=float)
    native_values = np.asarray(native_events[native_name], dtype=float)
    gpu_times = np.asarray(gpu_events["times"], dtype=float)
    gpu_values = np.asarray(gpu_events[gpu_name], dtype=float)

    mask = (gpu_times >= native_times[0]) & (gpu_times <= native_times[-1])
    aligned_native = np.interp(gpu_times[mask], native_times, native_values)
    aligned_gpu = gpu_values[mask]
    diff = np.abs(aligned_native - aligned_gpu)
    finite = np.isfinite(aligned_native) & np.isfinite(aligned_gpu)
    assert np.all(finite), f"{gpu_name}: non-finite values in compared traces"
    max_idx = int(np.argmax(diff))
    assert np.allclose(aligned_native, aligned_gpu, atol=atol, rtol=0.0), (
        f"{gpu_name}: max_abs_diff={diff[max_idx]} at t={gpu_times[mask][max_idx]}, "
        f"native={aligned_native[max_idx]}, gpu={aligned_gpu[max_idx]}, atol={atol}"
    )


def compare_conductance(native_events, gpu_events, native_r_name, native_d_name, gpu_name, atol):
    native_times = np.asarray(native_events["times"], dtype=float)
    native_values = np.asarray(native_events[native_r_name], dtype=float) + np.asarray(native_events[native_d_name], dtype=float)
    gpu_times = np.asarray(gpu_events["times"], dtype=float)
    gpu_values = np.asarray(gpu_events[gpu_name], dtype=float)

    mask = (gpu_times >= native_times[0]) & (gpu_times <= native_times[-1])
    aligned_native = np.interp(gpu_times[mask], native_times, native_values)
    aligned_gpu = gpu_values[mask]
    diff = np.abs(aligned_native - aligned_gpu)
    finite = np.isfinite(aligned_native) & np.isfinite(aligned_gpu)
    assert np.all(finite), f"{gpu_name}: non-finite values in compared traces"
    max_idx = int(np.argmax(diff))
    assert np.allclose(aligned_native, aligned_gpu, atol=atol, rtol=0.0), (
        f"{gpu_name}: max_abs_diff={diff[max_idx]} at t={gpu_times[mask][max_idx]}, "
        f"native={aligned_native[max_idx]}, gpu={aligned_gpu[max_idx]}, atol={atol}"
    )


def native_conductance(events, receptor_name):
    return np.asarray(events[f"g_r_{receptor_name}_1"]) + np.asarray(events[f"g_d_{receptor_name}_1"])


def plot_cm_default_comparison(native, gpu, output_dir):
    if not TEST_PLOTS:
        print(f"Skipping GPU compartmental comparison plots: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing GPU compartmental comparison plots to {output_dir}")
    w_legends = False

    plt.figure("gpu compartmental model test - voltage", figsize=(6, 6))
    ax_soma = plt.subplot(221)
    ax_soma.set_title("NEST")
    ax_soma.plot(native["passive"]["times"], native["passive"]["v_comp0"], c="b", label="passive dend")
    ax_soma.plot(native["active"]["times"], native["active"]["v_comp0"], c="b", ls="--", lw=2.0, label="active dend")
    ax_soma.set_xlabel(r"$t$ (ms)")
    ax_soma.set_ylabel(r"$v_{soma}$ (mV)")
    ax_soma.set_ylim((-90.0, 40.0))
    if w_legends:
        ax_soma.legend(loc=0)

    ax_dend = plt.subplot(222)
    ax_dend.set_title("NEST")
    ax_dend.plot(native["passive"]["times"], native["passive"]["v_comp1"], c="r", label="passive dend")
    ax_dend.plot(native["active"]["times"], native["active"]["v_comp1"], c="r", ls="--", lw=2.0, label="active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"$v_{dend}$ (mV)")
    ax_dend.set_ylim((-90.0, 40.0))
    if w_legends:
        ax_dend.legend(loc=0)

    ax_soma = plt.subplot(223)
    ax_soma.set_title("NEST-GPU NESTML")
    ax_soma.plot(gpu["passive"]["times"], gpu["passive"]["v_comp0"], c="b", label="passive dend")
    ax_soma.plot(gpu["active"]["times"], gpu["active"]["v_comp0"], c="b", ls="--", lw=2.0, label="active dend")
    ax_soma.set_xlabel(r"$t$ (ms)")
    ax_soma.set_ylabel(r"$v_{soma}$ (mV)")
    ax_soma.set_ylim((-90.0, 40.0))
    if w_legends:
        ax_soma.legend(loc=0)

    ax_dend = plt.subplot(224)
    ax_dend.set_title("NEST-GPU NESTML")
    ax_dend.plot(gpu["passive"]["times"], gpu["passive"]["v_comp1"], c="r", label="passive dend")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["v_comp1"], c="r", ls="--", lw=2.0, label="active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"$v_{dend}$ (mV)")
    ax_dend.set_ylim((-90.0, 40.0))
    if w_legends:
        ax_dend.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_compartmental_model_test - voltage.png"))
    plt.close()

    plt.figure("gpu compartmental model test - channel state variables", figsize=(6, 6))
    ax_soma = plt.subplot(221)
    ax_soma.set_title("NEST")
    ax_soma.plot(native["passive"]["times"], native["passive"]["m_Na_0"], c="b", label="m_Na passive dend")
    ax_soma.plot(native["passive"]["times"], native["passive"]["h_Na_0"], c="r", label="h_Na passive dend")
    ax_soma.plot(native["passive"]["times"], native["passive"]["n_K_0"], c="g", label="n_K passive dend")
    ax_soma.plot(native["active"]["times"], native["active"]["m_Na_0"], c="b", ls="--", lw=2.0, label="m_Na active dend")
    ax_soma.plot(native["active"]["times"], native["active"]["h_Na_0"], c="r", ls="--", lw=2.0, label="h_Na active dend")
    ax_soma.plot(native["active"]["times"], native["active"]["n_K_0"], c="g", ls="--", lw=2.0, label="n_K active dend")
    ax_soma.set_xlabel(r"$t$ (ms)")
    ax_soma.set_ylabel(r"svar")
    ax_soma.set_ylim((0.0, 1.0))
    if w_legends:
        ax_soma.legend(loc=0)

    ax_dend = plt.subplot(222)
    ax_dend.set_title("NEST")
    ax_dend.plot(native["passive"]["times"], native["passive"]["m_Na_1"], c="b", label="m_Na passive dend")
    ax_dend.plot(native["passive"]["times"], native["passive"]["h_Na_1"], c="r", label="h_Na passive dend")
    ax_dend.plot(native["passive"]["times"], native["passive"]["n_K_1"], c="g", label="n_K passive dend")
    ax_dend.plot(native["active"]["times"], native["active"]["m_Na_1"], c="b", ls="--", lw=2.0, label="m_Na active dend")
    ax_dend.plot(native["active"]["times"], native["active"]["h_Na_1"], c="r", ls="--", lw=2.0, label="h_Na active dend")
    ax_dend.plot(native["active"]["times"], native["active"]["n_K_1"], c="g", ls="--", lw=2.0, label="n_K active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"svar")
    ax_dend.set_ylim((0.0, 1.0))
    if w_legends:
        ax_dend.legend(loc=0)

    ax_soma = plt.subplot(223)
    ax_soma.set_title("NEST-GPU NESTML")
    ax_soma.plot(gpu["passive"]["times"], gpu["passive"]["m_Na0"], c="b", label="m_Na passive dend")
    ax_soma.plot(gpu["passive"]["times"], gpu["passive"]["h_Na0"], c="r", label="h_Na passive dend")
    ax_soma.plot(gpu["passive"]["times"], gpu["passive"]["n_K0"], c="g", label="n_K passive dend")
    ax_soma.plot(gpu["active"]["times"], gpu["active"]["m_Na0"], c="b", ls="--", lw=2.0, label="m_Na active dend")
    ax_soma.plot(gpu["active"]["times"], gpu["active"]["h_Na0"], c="r", ls="--", lw=2.0, label="h_Na active dend")
    ax_soma.plot(gpu["active"]["times"], gpu["active"]["n_K0"], c="g", ls="--", lw=2.0, label="n_K active dend")
    ax_soma.set_xlabel(r"$t$ (ms)")
    ax_soma.set_ylabel(r"svar")
    ax_soma.set_ylim((0.0, 1.0))
    if w_legends:
        ax_soma.legend(loc=0)

    ax_dend = plt.subplot(224)
    ax_dend.set_title("NEST-GPU NESTML")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["m_Na1"], c="b", ls="--", lw=2.0, label="m_Na active dend")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["h_Na1"], c="r", ls="--", lw=2.0, label="h_Na active dend")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["n_K1"], c="g", ls="--", lw=2.0, label="n_K active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"svar")
    ax_dend.set_ylim((0.0, 1.0))
    if w_legends:
        ax_dend.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_compartmental_model_test - channel state variables.png"))
    plt.close()

    plt.figure("gpu compartmental model test - dendritic synapse conductances", figsize=(3, 6))
    ax_dend = plt.subplot(211)
    ax_dend.set_title("NEST")
    ax_dend.plot(native["passive"]["times"], native_conductance(native["passive"], "AN_AMPA"), c="b", label="AMPA passive dend")
    ax_dend.plot(native["passive"]["times"], native_conductance(native["passive"], "AN_NMDA"), c="r", label="NMDA passive dend")
    ax_dend.plot(native["active"]["times"], native_conductance(native["active"], "AN_AMPA"), c="b", ls="--", lw=2.0, label="AMPA active dend")
    ax_dend.plot(native["active"]["times"], native_conductance(native["active"], "AN_NMDA"), c="r", ls="--", lw=2.0, label="NMDA active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"$g_{syn1}$ (uS)")
    if w_legends:
        ax_dend.legend(loc=0)

    ax_dend = plt.subplot(212)
    ax_dend.set_title("NEST-GPU NESTML")
    ax_dend.plot(gpu["passive"]["times"], gpu["passive"]["g_AN_AMPA1"], c="b", label="AMPA passive dend")
    ax_dend.plot(gpu["passive"]["times"], gpu["passive"]["g_AN_NMDA1"], c="r", label="NMDA passive dend")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["g_AN_AMPA1"], c="b", ls="--", lw=2.0, label="AMPA active dend")
    ax_dend.plot(gpu["active"]["times"], gpu["active"]["g_AN_NMDA1"], c="r", ls="--", lw=2.0, label="NMDA active dend")
    ax_dend.set_xlabel(r"$t$ (ms)")
    ax_dend.set_ylabel(r"$g_{syn1}$ (uS)")
    if w_legends:
        ax_dend.legend(loc=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_compartmental_model_test - dendritic synapse conductances.png"))
    plt.close()


class TestNESTGPUCompartmentalModel:
    def test_cm_default_against_native_nest(self, tmp_path):
        tests_path = os.path.realpath(os.path.dirname(__file__))
        target_path = os.path.join(tests_path, "target")
        nest_gpu_path = os.environ.get("NEST_GPU", os.getcwd())
        assert os.path.isdir(os.path.join(nest_gpu_path, "src"))

        generate_gpu_default_model(target_path)
        native = run_native_cm_default()
        gpu = run_gpu_cm_default(tmp_path)
        plot_cm_default_comparison(native, gpu, target_path)

        for section in ("passive", "active"):
            compare_trace(native[section], gpu[section], "v_comp0", "v_comp0", atol=2.0)
            compare_trace(native[section], gpu[section], "v_comp1", "v_comp1", atol=0.5)
            compare_trace(native[section], gpu[section], "m_Na_0", "m_Na0", atol=0.02)
            compare_trace(native[section], gpu[section], "h_Na_0", "h_Na0", atol=0.01)
            compare_trace(native[section], gpu[section], "n_K_0", "n_K0", atol=0.01)
            compare_conductance(native[section], gpu[section], "g_r_AN_AMPA_1", "g_d_AN_AMPA_1", "g_AN_AMPA1", atol=0.03)
            compare_conductance(native[section], gpu[section], "g_r_AN_NMDA_1", "g_d_AN_NMDA_1", "g_AN_NMDA1", atol=0.03)

        compare_trace(native["active"], gpu["active"], "m_Na_1", "m_Na1", atol=0.02)
        compare_trace(native["active"], gpu["active"], "h_Na_1", "h_Na1", atol=0.01)
        compare_trace(native["active"], gpu["active"], "n_K_1", "n_K1", atol=0.01)
