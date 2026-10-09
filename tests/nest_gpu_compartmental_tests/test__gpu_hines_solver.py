"""Check the extracted production Hines plan and arithmetic on the host.

CUDA compilation is covered by test__gpu_tree_solver_options. These numerical
checks run the same scalar routines called by the generated CUDA kernels.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
import pytest


HARNESS = r"""
void adjacency(const std::vector<int>& parents, std::vector<int>& offsets,
               std::vector<int>& children)
{
  offsets.assign(parents.size() + 1, 0);
  for (const int parent : parents)
  {
    if (parent >= static_cast<int>(parents.size()))
      throw std::runtime_error("Invalid parent");
    if (parent >= 0) ++offsets[parent + 1];
  }
  for (size_t i = 1; i < offsets.size(); ++i) offsets[i] += offsets[i - 1];
  children.resize(offsets.back());
  std::vector<int> next = offsets;
  for (size_t i = 0; i < parents.size(); ++i)
    if (parents[i] >= 0) children[next[parents[i]]++] = static_cast<int>(i);
}

extern "C" int hines_validate(const int* parent_data, int size, int compartments)
{
  try
  {
    std::vector<int> parents(parent_data, parent_data + size), offsets, children;
    adjacency(parents, offsets, children);
    cm_tree_solver_test::validate_topology(parents, offsets, children, compartments);
    return 0;
  }
  catch (const std::exception&) { return 1; }
}

extern "C" int hines_solve(const int* parent_data, int size, int compartments,
                           float* diagonal, float* rhs, const float* coupling,
                           float* vars, int n_var, int voltage_base,
                           const float* thresholds, int* spikes, bool* above)
{
  try
  {
    std::vector<int> parents(parent_data, parent_data + size), offsets, children;
    adjacency(parents, offsets, children);
    cm_hines_test::Plan plan;
    plan.build(parents, offsets, children, compartments);
    for (size_t level = 1; level + 1 < plan.level_offsets.size(); ++level)
      for (int i = plan.level_offsets[level]; i < plan.level_offsets[level + 1]; ++i)
        cm_hines_test::downsweep_node(plan.downsweep_nodes[i], offsets.data(),
                                     children.data(), coupling, diagonal, rhs);
    for (int neuron = 0; neuron < size / compartments; ++neuron)
      cm_hines_test::solve_root_and_upsweep(neuron, vars, n_var, voltage_base,
        diagonal, rhs, coupling, parents.data(), compartments, thresholds, spikes, above);
    return 0;
  }
  catch (const std::exception&) { return 1; }
}
"""


@pytest.fixture(scope="module")
def hines(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    templates = Path(__file__).resolve().parents[2] / "pynestml/codegeneration/resources_nest_gpu_compartmental/cm_neuron"
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined)
    source = "\n".join(env.get_template(f"hines/{part}").render(cm_unique_suffix="_test")
                       for part in ("plan.h.jinja2", "numerics.h.jinja2"))
    target = tmp_path_factory.mktemp("hines_numerics")
    source_path = target / "solver.cpp"
    source_path.write_text(source + HARNESS)
    library_path = target / "solver.so"
    result = subprocess.run([compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
                             "-shared", "-fPIC", str(source_path), "-o", str(library_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    library = ctypes.CDLL(str(library_path))
    array = np.ctypeslib.ndpointer
    library.hines_validate.argtypes = [array(np.int32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]
    library.hines_validate.restype = ctypes.c_int
    library.hines_solve.argtypes = [array(np.int32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
                                   *[array(np.float32, flags="C_CONTIGUOUS")] * 4,
                                   ctypes.c_int, ctypes.c_int, array(np.float32, flags="C_CONTIGUOUS"),
                                   array(np.int32, flags="C_CONTIGUOUS"), array(np.bool_, flags="C_CONTIGUOUS")]
    library.hines_solve.restype = ctypes.c_int
    return library


@pytest.mark.parametrize("kind", ["chain", "star", "binary", "mixed"])
@pytest.mark.parametrize("compartments", [1, 2, 9, 65])
def test_hines_against_dense_reference(hines, kind, compartments):
    rng = np.random.default_rng(100 + compartments)
    neurons = 3
    size = neurons * compartments
    parents = np.full(size, -1, dtype=np.int32)
    for neuron in range(neurons):
        base = neuron * compartments
        for local in range(1, compartments):
            shape = ("chain", "star", "binary")[neuron] if kind == "mixed" else kind
            parent = local - 1 if shape == "chain" else 0 if shape == "star" else (local - 1) // 2
            parents[base + local] = base + parent
    coupling = -rng.uniform(0.01, 2.0, size).astype(np.float32)
    coupling[parents < 0] = 0
    n_var, voltage_base = compartments + 3, 2
    variables = np.full((neurons, n_var), -123.0, dtype=np.float32)
    spikes = np.zeros(neurons, dtype=np.int32)
    above = np.zeros(neurons, dtype=np.bool_)
    # Refresh timestep coefficients and RHS, as matrix construction does. Check
    # threshold history survives external assignments to the recorded voltages.
    for step in range(4):
        diagonal = rng.uniform(1.0, 3.0, size).astype(np.float32)
        for child, parent in enumerate(parents):
            if parent >= 0:
                diagonal[child] -= coupling[child]
                diagonal[parent] -= coupling[child]
        rhs = rng.uniform(-2.0, 2.0, size).astype(np.float32)
        expected = []
        for neuron in range(neurons):
            base = neuron * compartments
            matrix = np.diag(diagonal[base:base + compartments].astype(np.float64))
            for local in range(1, compartments):
                parent = parents[base + local] - base
                matrix[local, parent] = matrix[parent, local] = coupling[base + local]
            expected.append(np.linalg.solve(matrix, rhs[base:base + compartments].astype(np.float64)))
        expected = np.asarray(expected)
        thresholds = (expected[:, 0] + (0.5 if step == 2 else -0.5)).astype(np.float32)
        variables[:, voltage_base:voltage_base + compartments] = -123.0
        assert hines.hines_solve(parents, size, compartments, diagonal, rhs, coupling,
                                 variables, n_var, voltage_base, thresholds, spikes, above) == 0
        np.testing.assert_allclose(variables[:, voltage_base:voltage_base + compartments],
                                   expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(spikes, np.full(neurons, int(step in (0, 3)), dtype=np.int32))
        np.testing.assert_array_equal(above, np.full(neurons, step != 2))
        np.testing.assert_array_equal(variables[:, :voltage_base], -123.0)
        np.testing.assert_array_equal(variables[:, -1], -123.0)


@pytest.mark.parametrize("parents,compartments", [
    ([-1, 2, 1], 3),                    # disconnected cycle
    ([-1, 1], 2),                       # self-parent
    ([-1, -1], 2),                      # second root
    ([-1, 0, -1, 0], 2),                # edge crosses neuron boundaries
    ([-1, 0, 1], 2),                    # unequal neuron sizes
])
def test_shared_topology_validation_rejects_invalid_forests(hines, parents, compartments):
    parents = np.asarray(parents, dtype=np.int32)
    assert hines.hines_validate(parents, len(parents), compartments) == 1
