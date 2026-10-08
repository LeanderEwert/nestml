"""Check that compartmental soma spikes enter NEST-GPU's event system."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target


HERE = Path(__file__).parent


def run_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module", params=["hines", "r_edd"])
def generated_model(request, tmp_path_factory):
    solver = request.param
    tmp = tmp_path_factory.mktemp("outgoing_" + solver)
    model = "cm_outgoing_" + solver
    source = tmp / (model + ".nestml")
    source.write_text((HERE / "resources/cm_ampa_only.nestml").read_text().replace("cm_ampa_only", model))
    gpu = tmp / "gpu"
    (gpu / "src").mkdir(parents=True)
    (gpu / "src/CMakeLists.txt").write_text("# <<BEGIN_NESTML_GENERATED>>\n# <<END_NESTML_GENERATED>>\n")
    target = tmp / "generated"
    generate_nest_gpu_compartmental_target(
        input_path=str(source), target_path=str(target), module_name="outgoing_module",
        suffix="_nestml", logging_level="ERROR", dev=True,
        codegen_opts={"gpu_tree_solver": solver, "nest_gpu_path": str(gpu),
                      "skip_build": True, "register_neuron_model": False, "nest_version": "v3"},
    )
    return {"solver": solver, "model": model + "_nestml", "source": source, "target": target}


def test_outgoing_cuda_compiles(generated_model):
    nvcc = shutil.which("nvcc")
    src = Path(os.environ.get("NEST_GPU", "")) / "src"
    if not nvcc or not (src / "spike_buffer.h").is_file():
        pytest.skip("nvcc and NEST_GPU/src are required for CUDA compilation")
    target, model = generated_model["target"], generated_model["model"]
    (target / "config.h").write_text("// Standalone compilation without MPI.\n")
    model_enum = f"i_{model}_model"
    enum_option = [] if model_enum in (src / "neuron_models.h").read_text() else [f"-D{model_enum}=0"]
    for name in (f"cm_tree_{model}.cu", f"{model}.cu"):
        # Supply an enum value if the isolated model has not already been
        # registered by an earlier integration-test run.
        run_checked([nvcc, "-std=c++17", "-diag-suppress", "177", "-I" + str(target),
                     "-I" + str(src), *enum_option, "-dc", str(target / name),
                     "-o", str(target / (name + ".o"))])


@pytest.fixture(scope="module")
def installed_model(generated_model):
    smi = shutil.which("nvidia-smi")
    if not smi or subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True).returncode:
        pytest.skip("A working NVIDIA driver is required for NEST-GPU simulation")
    if not (Path(os.environ.get("NEST_GPU", "")) / "src/CMakeLists.txt").is_file():
        pytest.skip("NEST_GPU must point to a NEST-GPU source tree")
    generate_nest_gpu_compartmental_target(
        input_path=str(generated_model["source"]),
        target_path=str(generated_model["target"].parent / "installed"),
        module_name="outgoing_module", suffix="_nestml", logging_level="ERROR", dev=True,
        codegen_opts={"gpu_tree_solver": generated_model["solver"]},
    )
    return generated_model["model"]


@pytest.mark.parametrize("profiling", [False, True], ids=["graph", "profiling"])
def test_spike_recording_delivery_and_stdp(installed_model, profiling, tmp_path):
    env = os.environ.copy()
    env.pop("NESTML_GPU_CM_PROFILE", None)
    if profiling:
        env["NESTML_GPU_CM_PROFILE"] = "1"
    output = tmp_path / "spikes.json"
    run_checked([sys.executable, str(HERE / "gpu_compartmental_outgoing_spikes_runner.py"),
                 installed_model, str(output)], env=env)
    result = json.loads(output.read_text())
    assert result["first_node"] > 0
    assert [len(times) for times in result["spikes"]] == [1, 0, 1, 1]
    assert [count[0] for count in result["counts"]] == [1, 0, 1, 1]
    assert [len(times) for times in result["downstream_spikes"]] == [1, 0, 1]
    for index in (0, 2):
        delay = result["downstream_spikes"][index][0] - result["spikes"][index][0]
        assert delay == pytest.approx(.5, abs=.10001)
    # All cells received the pre-spike. Only the cells that emitted a post-spike
    # should potentiate, including the last cell, which has no outgoing connection.
    for index in (0, 2, 3):
        assert result["weights"][index] > 10.1
    assert result["weights"][1] == pytest.approx(10., abs=1e-5)


@pytest.mark.parametrize("profiling", [False, True], ids=["graph", "profiling"])
def test_external_voltage_threshold_memory(installed_model, profiling, tmp_path):
    env = os.environ.copy()
    env.pop("NESTML_GPU_CM_PROFILE", None)
    if profiling:
        env["NESTML_GPU_CM_PROFILE"] = "1"
    output = tmp_path / "threshold.json"
    run_checked([sys.executable, str(HERE / "gpu_compartmental_stdp_runner.py"),
                 installed_model, str(output), "--threshold-only"], env=env)
    snapshots = json.loads(output.read_text())
    assert [len(times) for times in snapshots] == [1, 1, 1, 2]
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert snapshots[3][0] == snapshots[0][0]
    assert snapshots[3][1] > snapshots[3][0]
