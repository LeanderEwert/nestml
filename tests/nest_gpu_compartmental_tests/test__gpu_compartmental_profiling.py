# -*- coding: utf-8 -*-
#
# test__gpu_compartmental_profiling.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

"""Fixed-size GPU workloads matching the benchmarks without recording.

Select one test when profiling and set NESTML_GPU_CM_SKIP_REBUILD=1 to use
the installed model. Simulation runs in a child process, just as in the
benchmarks, so the profiler must include child processes.
Set NESTML_GPU_CM_PROFILE_SIZE to override the default 4096 neurons or added
dendritic compartments (also set by the profiling script's --size option).
"""

import math
import os
import sys

import pytest

TESTS_PATH = os.path.realpath(os.path.dirname(__file__))
if TESTS_PATH not in sys.path:
    sys.path.insert(0, TESTS_PATH)

from test__gpu_compartmental_population_benchmark import (  # noqa: E402
    DT,
    SIM_TIME,
    run_gpu_active_compartment_cm_default,
    run_gpu_active_population_cm_default,
)
from test__gpu_compartmental_population_benchmark import benchmark_target_path  # noqa: E402, F401


try:
    PROFILE_SIZE = int(os.environ.get("NESTML_GPU_CM_PROFILE_SIZE", 2 ** 12))
except ValueError as exc:
    raise ValueError("NESTML_GPU_CM_PROFILE_SIZE must be a positive integer") from exc
if PROFILE_SIZE <= 0:
    raise ValueError("NESTML_GPU_CM_PROFILE_SIZE must be a positive integer")


def print_profiling_header(setup):
    print(f"\n=== NEST-GPU profiling: {setup} ===", flush=True)
    print(f"duration={SIM_TIME} ms, resolution={DT} ms, recording=off, "
          "one GPU run, no CPU reference", flush=True)


def check_profiling_result(result):
    assert result["recording_enabled"] is False
    assert math.isfinite(result["runtime"]) and result["runtime"] > 0
    print(f"Profiling run finished: {result['runtime']:.6f} s "
          "(runner wall time, including setup)", flush=True)


@pytest.mark.usefixtures("benchmark_target_path")
class TestNESTGPUCompartmentalProfiling:
    def test_population(self, tmp_path):
        print_profiling_header(
            f"{PROFILE_SIZE} active neurons, two compartments and one somatic AMPA receptor per neuron")
        result = run_gpu_active_population_cm_default(
            tmp_path, PROFILE_SIZE, sample_neuron=0, record=False)
        assert result["n_neurons"] == PROFILE_SIZE
        check_profiling_result(result)

    def test_linear_compartments(self, tmp_path):
        print_profiling_header(
            f"one active chain, soma + {PROFILE_SIZE} dendritic compartments, "
            "one somatic AMPA receptor")
        result = run_gpu_active_compartment_cm_default(
            tmp_path, PROFILE_SIZE, sample_compartment=PROFILE_SIZE,
            record=False, morphology="chain")
        assert result["n_compartments"] == PROFILE_SIZE + 1
        assert result["morphology"] == "chain"
        check_profiling_result(result)

    def test_star_compartments(self, tmp_path):
        print_profiling_header(
            f"one passive star, soma + {PROFILE_SIZE} dendritic compartments, "
            "one somatic AMPA receptor")
        result = run_gpu_active_compartment_cm_default(
            tmp_path, PROFILE_SIZE, sample_compartment=PROFILE_SIZE,
            record=False, morphology="star")
        assert result["n_compartments"] == PROFILE_SIZE + 1
        assert result["morphology"] == "star"
        check_profiling_result(result)
