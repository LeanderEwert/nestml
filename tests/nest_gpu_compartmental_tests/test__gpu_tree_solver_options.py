"""Codegen selection checks that do not modify a NEST-GPU installation."""
from pathlib import Path

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
        "::enqueue_population_update", 1)[0]
    if solver == "r_edd":
        assert "r_edd_solver_->solve(" in solve_body
        assert "cm_tree_downsweep_level_kernel<<<" not in solve_body
        assert "r_edd_solver_->initialize(parent_index, child_offsets, child_indices, hh," in source
        assert "plan.build(parents, child_offsets, child_indices, compartments_per_neuron," in source
        assert "compartments_per_neuron,\n             4, 7);" in source
        assert "::Solver> r_edd_solver_;" in header
        # Numerical launches, including the base case, use the supplied stream.
        numerical_solve = source.split("void Solver::solve(", 1)[1].split("namespace", 1)[0]
        assert "base_kernel<<<" in numerical_solve
        assert "cudaMemcpy" not in numerical_solve
        assert "cudaMalloc" not in numerical_solve
    else:
        assert "cm_tree_downsweep_level_kernel<<<" in solve_body
        assert "cm_tree_solve_root_and_upsweep_kernel<<<" in solve_body
        assert "r_edd" not in source
        assert "r_edd" not in header
