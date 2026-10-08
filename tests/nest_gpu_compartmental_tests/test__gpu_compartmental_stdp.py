"""GPU counterpart of nest_compartmental_tests/test__compartmental_stdp.py.

Compare built-in NEST-GPU STDP on an ordinary receptor with cogenerated STDP.
The native model uses nearest-neighbour pairing; the isolated pair followed by
one measuring pre-spike used here also applies to the accumulating-trace model.
Set NESTML_GPU_CM_STDP_PLOTS=1 to save the CPU test's four-panel diagnostic plot.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target


HERE = Path(__file__).parent
ROOT = HERE.parents[1]
INPUTS = [str(HERE.parent / "nest_compartmental_tests/resources/concmech.nestml"),
          str(ROOT / "models/synapses/stdp_synapse.nestml")]
# Preserve the synapse suffix used by the CPU test, but give the generated neuron
# a distinct name to avoid replacing models used by other GPU tests.
MODEL = "multichannel_stdp_test_nestml"
RUNNER = HERE / "gpu_compartmental_stdp_runner.py"
OPTIONS = {"neuron_synapse_pairs": [{"neuron": "multichannel_stdp_test",
            "synapses": {"stdp_synapse": {"post_ports": ["post_spikes"]}}}],
           "weight_variable": {"stdp_synapse": "w"}}


def run_checked(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def generated_stdp(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gpu_stdp")
    source = tmp / "concmech.nestml"
    source.write_text(Path(INPUTS[0]).read_text().replace("multichannel_test_model", "multichannel_stdp_test"))
    inputs = [str(source), INPUTS[1]]
    gpu = tmp / "gpu"
    (gpu / "src").mkdir(parents=True)
    (gpu / "src/CMakeLists.txt").write_text("# <<BEGIN_NESTML_GENERATED>>\n# <<END_NESTML_GENERATED>>\n")
    target = tmp / "generated"
    generate_nest_gpu_compartmental_target(
        input_path=inputs, target_path=str(target), module_name="gpu_stdp_module",
        suffix="_nestml", logging_level="ERROR", dev=True,
        codegen_opts={**OPTIONS, "nest_gpu_path": str(gpu), "register_neuron_model": False,
                      "skip_build": True, "nest_version": "v3"},
    )
    return target, inputs


def test_stdp_cuda_compiles(generated_stdp):
    nvcc = shutil.which("nvcc")
    src = Path(os.environ.get("NEST_GPU", "")) / "src"
    if not nvcc or not (src / "spike_buffer.h").is_file():
        pytest.skip("nvcc and NEST_GPU/src are required for CUDA compilation")
    target, _ = generated_stdp
    (target / "config.h").write_text("// Standalone compilation without MPI.\n")
    model_enum = f"i_{MODEL}_model"
    enum_option = [] if model_enum in (src / "neuron_models.h").read_text() else [f"-D{model_enum}=0"]
    # Compile every translation unit: concmech contains channels and concentration
    # mechanisms in addition to the ordinary and paired receptors used by the test.
    for source in sorted(target.glob("*.cu")):
        run_checked([nvcc, "-std=c++17", "-diag-suppress", "177", "-I" + str(target),
                     "-I" + str(src), *enum_option, "-dc", str(source),
                     "-o", str(source.with_suffix(".o"))])


@pytest.fixture(scope="module")
def installed_stdp(generated_stdp):
    smi = shutil.which("nvidia-smi")
    if not smi or subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                                  capture_output=True).returncode:
        pytest.skip("A working NVIDIA driver is required for NEST-GPU simulation")
    if not (Path(os.environ.get("NEST_GPU", "")) / "src/CMakeLists.txt").is_file():
        pytest.skip("NEST_GPU must point to a NEST-GPU source tree")
    target, inputs = generated_stdp
    generate_nest_gpu_compartmental_target(
        input_path=inputs, target_path=str(target.parent / "installed"), module_name="gpu_stdp_module",
        suffix="_nestml", logging_level="ERROR", dev=True, codegen_opts=OPTIONS,
    )
    return target


@pytest.fixture(scope="module")
def stdp_results(installed_stdp):
    target = installed_stdp
    env = os.environ.copy()
    env.pop("NESTML_GPU_CM_PROFILE", None)
    results = []
    for pre_time in range(1, 20):
        output = target.parent / f"stdp_pre_{pre_time}.json"
        run_checked([sys.executable, str(RUNNER), MODEL, str(output), "--pre-time", str(pre_time)], env=env)
        results.append(json.loads(output.read_text()))
    if os.environ.get("NESTML_GPU_CM_STDP_PLOTS") == "1":
        plot_results(results, target.parent / "gpu_compartmental_stdp.png")
    return results


def test_sub_timestep_delay_reports_error(installed_stdp):
    output = installed_stdp.parent / "invalid_delay.json"
    result = subprocess.run(
        [sys.executable, str(RUNNER), MODEL, str(output), "--pre-time", "1", "--receptor-delay", ".05"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Synaptic receptor delay must be at least one timestep" in result.stderr, result.stdout + result.stderr


def test_compartmental_stdp(stdp_results):
    differences = []
    for result in stdp_results:
        context = f"pre-post offset {result['offset']} ms"
        # One relay spike per stimulus, with the same relay/connection path into
        # both branches. Check timing before interpreting any weight difference.
        assert len(result["pre_spikes"]) == 2, context
        assert result["pre_spikes"][0] == pytest.approx(result["pre_time"], abs=.20001), context
        assert result["pre_spikes"][1] == pytest.approx(21., abs=.20001), context
        for name in ("native", "generated"):
            branch = result[name]
            assert len(branch["post_spikes"]) == 1, (context, name, branch["post_spikes"])
            assert branch["post_spikes"][0] == pytest.approx(10., abs=.10001), (context, name)
            assert np.isfinite(branch["weight"]), (context, name)
            if result["offset"] < 0:
                assert branch["weight"] > 10., (context, name, "missing potentiation")
            elif result["offset"] > 0:
                assert branch["weight"] < 10., (context, name, "missing depression")
            for trace in branch["events"].values():
                assert trace and np.all(np.isfinite(trace)), (context, name)
        assert result["native"]["post_spikes"] == pytest.approx(result["generated"]["post_spikes"], abs=1e-5), context
        events = result["generated"]["events"]
        assert max(events["pre_trace0"]) >= .99, context
        assert max(events["post_trace0"]) >= .99, context
        assert events["w0"][-1] == pytest.approx(result["generated"]["weight"], abs=1e-5), context
        native, generated = result["native"]["weight"], result["generated"]["weight"]
        print(f"{context}: native={native:.9g}, generated={generated:.9g}, diff={generated-native:.9g}")
        if result["offset"] != 0:
            differences.append((abs(generated - native), result["offset"]))
    error, offset = max(differences)
    assert error <= .01, f"Maximum weight difference {error} exceeds 0.01 at offset {offset} ms"


def plot_results(results, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    offsets = [result["offset"] for result in results]
    native = np.array([result["native"]["weight"] for result in results])
    generated = np.array([result["generated"]["weight"] for result in results])
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(offsets, native, "o", color="grey", label="NEST-GPU stdp")
    axes[0].plot(offsets, generated, "x", color="orange", label="generated STDP")
    axes[0].set_ylabel("Final weight")
    axes[0].legend()
    axes[1].vlines(offsets, 0, generated - native, color="red")
    axes[1].set_ylabel("Weight difference")
    for axis, name in zip(axes[2:], ("native", "generated")):
        for index, result in enumerate(results):
            for times, color, label in ((result["pre_spikes"], "blue", "pre"),
                                        (result[name]["post_spikes"], "red", "post")):
                axis.plot([result["offset"]] * len(times), times, "_", color=color,
                          label=label if index == 0 else None)
        axis.set_ylabel(f"{name} spikes (ms)")
        axis.legend()
    axes[-1].set_xlabel("Presynaptic stimulus − postsynaptic stimulus (ms)")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"STDP diagnostic plot: {output}")
