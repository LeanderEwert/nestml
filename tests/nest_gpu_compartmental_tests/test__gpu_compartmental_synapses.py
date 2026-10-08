"""Compile paired GPU receptors and compare NEST-GPU simulations with NEST."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pynestml.frontend.pynestml_frontend import (
    generate_nest_compartmental_target,
    generate_nest_gpu_compartmental_target,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = "cm_ampa_only_nestml"
SYNAPSES = ["stdp_synapse", "stdp_nn_symm_synapse", "third_factor_stdp_synapse"]
INPUTS = [str(ROOT / "tests/nest_gpu_compartmental_tests/resources/cm_ampa_only.nestml"),
          str(ROOT / "models/synapses/stdp_synapse.nestml"),
          str(ROOT / "models/synapses/stdp_nn_symm_synapse.nestml"),
          str(ROOT / "tests/nest_compartmental_tests/resources/cm_third_factor_stdp_synapse.nestml")]
RUNNER = Path(__file__).with_name("gpu_compartmental_synapse_runner.py")


def pairing_options():
    return {"neuron_synapse_pairs": [{"neuron": "cm_ampa_only",
            "synapses": {name: {"post_ports": ["post_spikes"]} for name in SYNAPSES}}],
            "weight_variable": {name: "w" for name in SYNAPSES}}


@pytest.fixture(scope="module")
def generated_synapses(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpu_synapses")
    gpu = tmp / "gpu"
    (gpu / "src").mkdir(parents=True)
    (gpu / "src/CMakeLists.txt").write_text("# <<BEGIN_NESTML_GENERATED>>\n# <<END_NESTML_GENERATED>>\n")
    target = tmp / "generated"
    generate_nest_gpu_compartmental_target(
        input_path=INPUTS, target_path=str(target), module_name="gpu_synapse_test_module",
        suffix="_nestml", logging_level="ERROR", dev=True,
        codegen_opts={**pairing_options(), "nest_gpu_path": str(gpu), "skip_build": True,
                      "register_neuron_model": False, "nest_version": "v3"},
    )
    return target


def run_checked(command):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_paired_cuda_compiles(generated_synapses):
    nvcc = shutil.which("nvcc")
    src = Path(os.environ.get("NEST_GPU", "")) / "src"
    if not nvcc or not (src / "nlohmann/json.hpp").is_file():
        pytest.skip("nvcc and NEST_GPU/src are required for CUDA compilation")
    target = generated_synapses
    # The generated tree uses cuda_error.h; MPI is irrelevant to this standalone compile.
    (target / "config.h").write_text("// Standalone compilation without MPI.\n")
    for name in [f"cm_group_receptor_currents_{MODEL}.cu", f"cm_tree_{MODEL}.cu"]:
        run_checked([nvcc, "-std=c++17", "-diag-suppress", "177", "-I" + str(target), "-I" + str(src),
                     "-dc", str(target / name), "-o", str(target / (name + ".o"))])


@pytest.fixture(scope="module")
def cpu_synapse_results(tmp_path_factory):
    pytest.importorskip("nest")
    tmp = tmp_path_factory.mktemp("cpu_synapses")
    install = tmp / "install"
    install.mkdir()
    generate_nest_compartmental_target(
        input_path=INPUTS, target_path=str(tmp / "generated"), install_path=str(install),
        module_name="cm_synapse_reference_module", suffix="_nestml", logging_level="ERROR",
        codegen_opts={**pairing_options(), "fp_precision": "single"},
    )
    output = tmp / "results.json"
    run_checked([sys.executable, str(RUNNER), "cpu", str(output),
                 str(install / "cm_synapse_reference_module.so")])
    return json.loads(output.read_text())


def assert_plasticity(results):
    for index, events in enumerate(results):
        for trace in events.values():
            assert np.all(np.isfinite(trace))
        for port in (0, 1):
            weights = np.asarray(events[f"w{port}"])
            assert weights.max() - weights[0] > .1  # causal potentiation
            assert weights[-1] < weights.max() - .01  # subsequent acausal depression
            assert max(events[f"pre_trace{port}"]) > .9
            assert max(events[f"post_trace{port}"]) > .9
        # Both plastic receptors receive the same input, but use different delays.
        first_response = np.asarray(events["times"]) < 14.
        times = np.asarray(events["times"])[first_response]
        peaks = [times[np.argmax(np.asarray(events[f"g_AMPA{port}"])[first_response])] for port in (0, 1)]
        assert peaks[1] - peaks[0] == pytest.approx(.3, abs=.11)
        third_weights = np.asarray(events["w3"])
        if index == 0:
            assert np.allclose(third_weights, third_weights[0], atol=1e-5)
        else:
            assert np.ptp(third_weights) > .1


def test_cpu_reference_protocol(cpu_synapse_results):
    assert_plasticity(cpu_synapse_results)


def test_gpu_synapses_match_cpu(tmp_path, cpu_synapse_results):
    smi = shutil.which("nvidia-smi")
    if not smi or subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True).returncode:
        pytest.skip("A working NVIDIA driver is required for NEST-GPU simulation")
    if not (Path(os.environ.get("NEST_GPU", "")) / "src/CMakeLists.txt").is_file():
        pytest.skip("NEST_GPU must point to a NEST-GPU source tree")
    # Follow the existing compartmental GPU tests: build/register, then simulate in
    # a fresh process so the newly compiled models are loaded by nestgpu.
    generate_nest_gpu_compartmental_target(
        input_path=INPUTS, target_path=str(tmp_path / "generated"), module_name="gpu_synapse_test_module",
        suffix="_nestml", logging_level="ERROR", dev=True, codegen_opts=pairing_options(),
    )
    output = tmp_path / "gpu.json"
    run_checked([sys.executable, str(RUNNER), "gpu", str(output)])
    gpu = json.loads(output.read_text())
    assert_plasticity(gpu)
    for reference, actual in zip(cpu_synapse_results, gpu):
        for port in (0, 1, 3):
            # Allow one timestep of threshold-crossing jitter between solvers.
            for name in (f"w{port}", f"pre_trace{port}", f"post_trace{port}"):
                assert actual[name][-1] == pytest.approx(reference[name][-1], rel=.01, abs=.01)
        for port in (0, 1):
            assert max(actual[f"g_AMPA{port}"]) == pytest.approx(max(reference[f"g_AMPA{port}"]), rel=.01)
