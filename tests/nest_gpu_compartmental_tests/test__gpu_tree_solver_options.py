"""Codegen selection checks that do not modify a NEST-GPU installation."""
from pathlib import Path
import shutil
import subprocess

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import pytest

from pynestml.codegeneration.nest_gpu_compartmental_code_generator import NESTGPUCompartmentalCodeGenerator
from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target


@pytest.mark.parametrize("option,value", [
    ("gpu_tree_solver", "unknown"), ("gpu_tree_solver", None),
    ("gpu_tree_solver_chain_length", 1), ("gpu_tree_solver_chain_length", True),
    ("gpu_tree_solver_chain_length", 3.0), ("gpu_tree_solver_base_size", 0),
    ("gpu_tree_solver_base_size", "32"), ("gpu_tree_solver_base_size", False),
])
def test_invalid_solver_options(option, value):
    with pytest.raises(ValueError, match=option):
        NESTGPUCompartmentalCodeGenerator({"nest_version": "v3", option: value})


def test_default_solver_options():
    generator = NESTGPUCompartmentalCodeGenerator({"nest_version": "v3"})
    assert generator.get_option("gpu_tree_solver") == "hines"
    assert generator.get_option("gpu_tree_solver_chain_length") == 3
    assert generator.get_option("gpu_tree_solver_base_size") == 32


@pytest.fixture(scope="module", params=[None, "hines", "r_edd"])
def generated_solver(request, tmp_path_factory):
    solver = request.param
    tmp_path = tmp_path_factory.mktemp("generated_tree")
    gpu = tmp_path / "nest_gpu"
    src = gpu / "src"
    src.mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("# <<BEGIN_NESTML_GENERATED>>\n# <<END_NESTML_GENERATED>>\n")
    target = tmp_path / "generated"
    options = dict(nest_gpu_path=str(gpu), register_neuron_model=False, skip_build=True, nest_version="v3")
    if solver is not None:
        options["gpu_tree_solver"] = solver
    if solver == "r_edd":
        options.update(gpu_tree_solver_chain_length=4, gpu_tree_solver_base_size=7)
    generate_nest_gpu_compartmental_target(
        input_path=str(Path(__file__).parent / "resources/cm_continuous_only.nestml"),
        target_path=str(target), module_name="r_edd_test_module", suffix="_nestml",
        logging_level="ERROR", dev=True, codegen_opts=options,
    )
    return solver, target


def test_generated_solver_selection(generated_solver):
    solver, target = generated_solver
    source = (target / "cm_tree_cm_continuous_only_nestml.cu").read_text()
    header = (target / "cm_tree_cm_continuous_only_nestml.h").read_text()
    solve_body = source.split("::enqueue_matrix_solve(cudaStream_t stream)", 1)[1].split(
        "::enqueue_spike_emission", 1)[0]
    assert "tree_solver_->solve(" in solve_body
    assert "<<<" not in solve_body
    assert "tree_solver_->initialize(parent_index, child_offsets, child_indices, hh," in source
    assert "std::shared_ptr<Solver> tree_solver_;" in header
    assert "validate_topology(" in source
    # The population tree no longer builds or uploads Hines execution arrays.
    population_init = source.split("::initialize_population_runtime(", 1)[1].split(
        "::enqueue_matrix_solve", 1)[0]
    for name in ("GPU_downsweep_nodes", "GPU_parent_index", "GPU_child_offsets", "GPU_child_indices",
                 "downsweep_level_offsets_", "remaining_children", "GPU_hh"):
        assert name not in population_init
    numerical_solve = source.split("void Solver::solve(", 1)[1].split("namespace", 1)[0]
    assert "cudaMemcpy" not in numerical_solve
    assert "cudaMalloc" not in numerical_solve
    assert "Synchronize" not in numerical_solve
    if solver == "r_edd":
        assert "plan.build(parents, child_offsets, child_indices, compartments_per_neuron," in source
        assert "compartments_per_neuron,\n             4, 7);" in source
        assert "using Solver = cm_r_eddCmContinuousOnlyNestml::Solver;" in header
        assert "downsweep_nodes" not in source
        assert "downsweep_nodes" not in header
        # Numerical launches, including the base case, use the supplied stream.
        assert "base_kernel<<<" in numerical_solve
    else:
        assert "using Solver = cm_hinesCmContinuousOnlyNestml::Solver;" in header
        assert "cm_tree_downsweep_level_kernel<<<" in numerical_solve
        assert "cm_tree_solve_root_and_upsweep_kernel<<<" in numerical_solve
        assert "0, stream>>>" in numerical_solve
        assert "r_edd" not in source
        assert "r_edd" not in header


@pytest.mark.parametrize("solver", ["hines", "r_edd"])
def test_solver_cuda_compiles(solver, tmp_path):
    compiler = shutil.which("nvcc")
    if compiler is None:
        pytest.skip("CUDA compiler unavailable")
    templates = Path(__file__).resolve().parents[2] / "pynestml/codegeneration/resources_nest_gpu_compartmental/cm_neuron"
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined)
    context = dict(cm_unique_suffix="_test", gpu_tree_solver_chain_length=3, gpu_tree_solver_base_size=2)
    source = tmp_path / "solver.cu"
    source.write_text("\n".join(env.get_template(f"{solver}/{part}").render(**context)
                               for part in ("device.h.jinja2", "kernels.cu.jinja2")))
    result = subprocess.run([compiler, "-std=c++14", "-arch=sm_80", "-c", str(source),
                             "-o", str(tmp_path / "solver.o")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
