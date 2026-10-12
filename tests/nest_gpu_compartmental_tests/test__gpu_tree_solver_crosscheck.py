# -*- coding: utf-8 -*-
#
# test__gpu_tree_solver_crosscheck.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) either version.
#

"""End-to-end cross-check of the two compartmental tree solvers on GPU.

The repo's NESTML source is generated once per solver (plain Hines and R-EDD)
and installed into NEST-GPU. Each solver is built and *its own scenarios run*
before the next solver is built, because both solvers install to the same
``libnestgpukernel.so`` and would otherwise overwrite each other. Each scenario
is executed by :mod:`gpu_compartmental_solver_runner` in its **own process**,
because NEST-GPU calibrates its neuron graph on the first ``Create`` and refuses
to create new nodes afterwards. The recorded voltage traces for the two solvers
are then compared to verify the refactor leaves results unchanged.

Each scenario is a purely passive cable-equation problem: the soma is
initialised off equilibrium and the resulting decay propagates through the
dendritic tree. No receptor or synapse is involved, so a disagreement points at
the tree solver itself rather than at any mechanism.
"""

import json
import os
import subprocess
import sys

import pytest
from test__gpu_compartmental_model import generate_gpu_default_model

tests_path = os.path.dirname(__file__)
sys.path.insert(0, tests_path)
import gpu_compartmental_solver_runner as runner_module

TEST_PLOTS = True
PLOT_IMPORT_ERROR = None
try:
    import matplotlib.pyplot as plt
except BaseException as e:
    TEST_PLOTS = False
    PLOT_IMPORT_ERROR = e

SOLVERS = ["hines", "r_edd"]
MORPHOLOGIES = ["chain", "star", "binary"]
COMPARTMENT_SIZES = [2, 4, 8]

# Soma initialised off equilibrium; the passive decay drives the fluctuations.
OFF_EQUILIBRIUM_VOLTAGE = runner_module.OFF_EQUILIBRIUM_VOLTAGE
DT = 0.1
SIM_TIME = 1000.0
ATOL = 1e-4

# Solvers are distinguished by line style (hines solid, r_edd dashed) so the two
# traces overlap when the check passes; the added-compartment count by colour.
SOLVER_STYLE = {"hines": "-", "r_edd": "--"}
SIZE_COLOR = {2: "tab:blue", 4: "tab:green", 8: "tab:orange"}


def _run_scenario(tmp_path, config, samples):
    """Run one scenario in a fresh process and return the parsed result JSON."""
    params = {
        "config": config,
        "sample_compartments": list(samples),
        "dt": DT,
        "sim_time": SIM_TIME,
    }
    params_path = tmp_path / "params.json"
    out_path = tmp_path / "result.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f)
    runner_script = os.path.join(tests_path, "gpu_compartmental_solver_runner.py")
    subprocess.check_call([sys.executable, runner_script, str(params_path), str(out_path)])
    return json.loads(out_path.read_text(encoding="utf-8"))


def _compare_trace(reference, gpu, reference_name, gpu_name, atol):
    """Raise unless every trace point of gpu matches reference within ``atol``."""
    assert len(reference) == len(gpu), (
        f"{reference_name} ({len(reference)}) and {gpu_name} ({len(gpu)}) have different lengths"
    )
    max_diff = max(abs(a - b) for a, b in zip(reference, gpu))
    assert max_diff <= atol, (
        f"{reference_name} vs {gpu_name}: max abs diff {max_diff:.3e} > ATOL {atol:.1e}"
    )


def plot_solver_comparison(gpu_results, output_dir):
    """Render the soma and leaf voltages of both solvers for each morphology.

    Each morphology produces one PNG with two subplots (soma ``v_comp0`` and the
    deepest compartment ``v_comp{n_added}``). Solvers are drawn with different
    line styles and the added-compartment count with different colours, so the
    hines and r_edd traces overlap when the cross-check passes and diverge only
    if a solver regressed.
    """
    if not TEST_PLOTS:
        print(f"Skipping GPU solver cross-check plots: matplotlib is unavailable ({PLOT_IMPORT_ERROR})")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing GPU solver cross-check plots to {output_dir}")

    for morphology in MORPHOLOGIES:
        plt.figure(f"solver crosscheck - {morphology}", figsize=(12, 5))
        ax_soma = plt.subplot(121)
        ax_soma.set_title(f"{morphology}: soma (v_comp0)")
        ax_soma.set_xlabel(r"$t$ (ms)")
        ax_soma.set_ylabel(r"$v_{soma}$ (mV)")

        ax_leaf = plt.subplot(122)
        ax_leaf.set_title(f"{morphology}: leaf (deepest compartment)")
        ax_leaf.set_xlabel(r"$t$ (ms)")
        ax_leaf.set_ylabel(r"$v_{leaf}$ (mV)")

        for solver in SOLVERS:
            for n_added in COMPARTMENT_SIZES:
                result = gpu_results.get((solver, morphology, n_added))
                if result is None:
                    continue
                traces = result["traces"]
                times = traces["times"]
                color = SIZE_COLOR[n_added]
                style = SOLVER_STYLE[solver]
                label = f"{solver} n_added={n_added}"
                ax_soma.plot(times, traces["v_comp0"], color=color, ls=style, lw=1.5, label=label)
                ax_leaf.plot(times, traces[f"v_comp{n_added}"], color=color, ls=style, lw=1.5, label=label)

        ax_soma.legend(loc=0, fontsize=7)
        ax_leaf.legend(loc=0, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"solver_crosscheck_{morphology}.png"))
        plt.close()


class TestGPUCompartmentalSolverCrosscheck:
    def test_hines_vs_r_edd(self, tmp_path_factory):
        os.environ.setdefault(
            "PYTHONPATH",
            "/home/levie/Desktop/HiWi/NEST-GPU/nest-gpu-install/lib/python3.12/site-packages",
        )

        plots_dir = tmp_path_factory.mktemp("plots")
        gpu_results = {}  # (solver, morphology, n_added) -> result dict

        # Build each solver and run ALL of its scenarios before the next build
        # overwrites the shared libnestgpukernel.so with the other solver's code.
        for solver in SOLVERS:
            target = tmp_path_factory.mktemp(f"target_solver_{solver}")
            generate_gpu_default_model(str(target), codegen_opts={"gpu_tree_solver": solver})

            for morphology in MORPHOLOGIES:
                for n_added in COMPARTMENT_SIZES:
                    samples = [0, 1, n_added]  # soma, first dendrite, deepest dendrite
                    assert all(0 <= s <= n_added for s in samples)
                    config = runner_module.build_neuron_configuration(
                        morphology, n_added, initial_voltage=OFF_EQUILIBRIUM_VOLTAGE)
                    tmp = tmp_path_factory.mktemp(f"run_{solver}_{morphology}_{n_added}")
                    result = _run_scenario(tmp, config, samples)
                    gpu_results[(solver, morphology, n_added)] = result
                    print(f"  {solver:6s} {morphology:7s} n_added={n_added:3d} "
                          f"runtime={result['runtime']:.4f}s")

        for morphology in MORPHOLOGIES:
            for n_added in COMPARTMENT_SIZES:
                samples = [0, 1, n_added]
                hines = gpu_results[("hines", morphology, n_added)]
                r_edd = gpu_results[("r_edd", morphology, n_added)]
                for index in samples:
                    hines_trace = hines["traces"][f"v_comp{index}"]
                    r_edd_trace = r_edd["traces"][f"v_comp{index}"]
                    _compare_trace(hines_trace, r_edd_trace,
                                   f"hines/{morphology}/v_comp{index}",
                                   f"r_edd/{morphology}/v_comp{index}", ATOL)
                print(f"  CROSSMATCH  {morphology:7s} n_added={n_added:3d}")

        plot_solver_comparison(gpu_results, str(plots_dir))
