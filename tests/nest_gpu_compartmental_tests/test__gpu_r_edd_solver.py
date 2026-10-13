# -*- coding: utf-8 -*-
#
# test__gpu_r_edd_solver.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) either version.
#

"""Check the R-EDD tree solver arithmetic against a dense host reference.

R-EDD (recursive exact domain decomposition) is a genuinely different algorithm
from plain Hines, so agreement between the two solvers (test__gpu_tree_solver
cross-check) does not by itself prove either one is correct. This test mirrors
the hines solver test: it compiles the exact scalar routines the production
R-EDD kernels call (plan.h + numerics.h) and runs them on the host, driving the
same gather -> chain/assemble -> base -> reconstruct -> scatter flow the CUDA
kernels use. The resulting voltages are compared with an independent dense
solve of the same passive cable system, so a disagreement points at the R-EDD
numerics rather than at the Hines solver or at the cross-check harness.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import numpy as np
import pytest


# The R-EDD numerics operate on a LevelView of pointers; the harness builds the
# same multi-level reduction hierarchy the kernels use, but on host memory.
HARNESS = r"""
namespace cm_r_edd_test
{
using cm_tree_solver_test::validate_topology;

struct HostLevel
{
  std::vector<int> parent, tree_offsets, junctions, junction_index;
  std::vector<int> chain_offsets, chain_nodes, chain_upper, chain_lower;
  std::vector<int> incident_offsets, incident_chains, parent_chain;
  std::vector<float> diagonal, rhs, coupling, upper_response, lower_response;
  int size, trees, chains, junction_count;

  LevelView view() const
  {
    return LevelView{parent.data(), tree_offsets.data(), junctions.data(),
                     chain_offsets.data(), chain_nodes.data(), chain_upper.data(), chain_lower.data(),
                     incident_offsets.data(), incident_chains.data(), parent_chain.data(),
                     const_cast<float*>(diagonal.data()), const_cast<float*>(rhs.data()),
                     const_cast<float*>(coupling.data()), const_cast<float*>(upper_response.data()),
                     const_cast<float*>(lower_response.data())};
  }
};

extern "C" int r_edd_solve(const int* parent_data, int size, int compartments,
                           float* diagonal, float* rhs, const float* coupling,
                           const float* thresholds, float* voltages,
                           int* spikes, bool* above)
{
  try
  {
    std::vector<int> parents(parent_data, parent_data + size);
    std::vector<int> child_offsets(size + 1, 0);
    for (int i = 0; i < size; ++i)
      if (parents[i] >= 0) ++child_offsets[parents[i] + 1];
    for (int i = 1; i <= size; ++i) child_offsets[i] += child_offsets[i - 1];
    std::vector<int> child_indices(child_offsets.back());
    std::vector<int> child_cursor = child_offsets;
    for (int i = 0; i < size; ++i)
      if (parents[i] >= 0) child_indices[child_cursor[parents[i]]++] = i;

    Plan plan;
    plan.build(parents, child_offsets, child_indices, compartments, 3, 2);
    validate_topology(parents, child_offsets, child_indices, compartments);

    std::vector<HostLevel> levels;
    levels.reserve(plan.levels.size());
    for (const auto& lp : plan.levels)
    {
      HostLevel hl;
      hl.parent = lp.parent;
      hl.tree_offsets = lp.tree_offsets;
      hl.junctions = lp.junctions;
      hl.junction_index = lp.junction_index;
      hl.chain_offsets = lp.chain_offsets;
      hl.chain_nodes = lp.chain_nodes;
      hl.chain_upper = lp.chain_upper;
      hl.chain_lower = lp.chain_lower;
      hl.incident_offsets = lp.incident_offsets;
      hl.incident_chains = lp.incident_chains;
      hl.parent_chain = lp.parent_chain;
      hl.size = static_cast<int>(lp.parent.size());
      hl.trees = static_cast<int>(lp.tree_offsets.size()) - 1;
      hl.chains = static_cast<int>(lp.chain_upper.size());
      hl.junction_count = static_cast<int>(lp.junctions.size());
      hl.diagonal.assign(hl.size, 0.0f);
      hl.rhs.assign(hl.size, 0.0f);
      hl.coupling.assign(hl.size, 0.0f);
      hl.upper_response.assign(hl.size, 0.0f);
      hl.lower_response.assign(hl.size, 0.0f);
      levels.push_back(std::move(hl));
    }

    std::vector<float> ordered_coupling(size);
    for (int node = 0; node < size; ++node)
    {
      const int perm = plan.permutation[node];
      levels[0].diagonal[node] = diagonal[perm];
      levels[0].rhs[node] = rhs[perm];
      ordered_coupling[node] = coupling[perm];
    }
    for (int node = 0; node < size; ++node) levels[0].coupling[node] = ordered_coupling[node];

    for (size_t depth = 0; depth + 1 < levels.size(); ++depth)
    {
      HostLevel& a = levels[depth];
      HostLevel& reduced = levels[depth + 1];
      for (int c = 0; c < a.chains; ++c) solve_chain(a.view(), c);
      for (int j = 0; j < a.junction_count; ++j) assemble_junction(a.view(), reduced.view(), j);
    }
    HostLevel& base = levels.back();
    for (int t = 0; t < base.trees; ++t) solve_base_tree(base.view(), t);
    for (int depth = static_cast<int>(levels.size()) - 2; depth >= 0; --depth)
    {
      HostLevel& a = levels[depth];
      HostLevel& reduced = levels[depth + 1];
      for (int c = 0; c < a.chains; ++c) reconstruct_chain(a.view(), reduced.view(), c);
      for (int j = 0; j < a.junction_count; ++j) a.rhs[a.junctions[j]] = reduced.rhs[j];
    }

    for (int node = 0; node < size; ++node)
    {
      const int original = plan.permutation[node];
      const int neuron = original / compartments;
      const int local = original % compartments;
      const float solved = levels[0].rhs[node];
      if (local == 0)
      {
        const bool above_threshold = solved >= thresholds[neuron];
        spikes[neuron] = above_threshold && !above[neuron];
        above[neuron] = above_threshold;
      }
      voltages[original] = solved;
    }
    return 0;
  }
  catch (const std::exception&) { return 1; }
}
}
"""


@pytest.fixture(scope="module")
def r_edd(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    templates = Path(__file__).resolve().parents[2] / "pynestml/codegeneration/resources_nest_gpu_compartmental/cm_neuron"
    env = Environment(loader=FileSystemLoader(templates), undefined=StrictUndefined)
    source = "\n".join(env.get_template(f"r_edd/{part}").render(cm_unique_suffix="_test")
                        for part in ("plan.h.jinja2", "numerics.h.jinja2"))
    # r_edd plan.h reuses the shared population topology but does not include it.
    source = env.get_template("solver/topology.h.jinja2").render(cm_unique_suffix="_test") + "\n" + source
    target = tmp_path_factory.mktemp("r_edd_numerics")
    source_path = target / "solver.cpp"
    source_path.write_text(source + "\n" + HARNESS)
    library_path = target / "solver.so"
    result = subprocess.run([compiler, "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
                             "-shared", "-fPIC", str(source_path), "-o", str(library_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    library = ctypes.CDLL(str(library_path))
    array = np.ctypeslib.ndpointer
    library.r_edd_solve.argtypes = [array(np.int32, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int,
                                    *[array(np.float32, flags="C_CONTIGUOUS")] * 5,
                                    array(np.int32, flags="C_CONTIGUOUS"), array(np.bool_, flags="C_CONTIGUOUS")]
    library.r_edd_solve.restype = ctypes.c_int
    return library


def _build_forest(kind, compartments, rng, neurons=3):
    parents = np.full(neurons * compartments, -1, dtype=np.int32)
    for neuron in range(neurons):
        base = neuron * compartments
        for local in range(1, compartments):
            shape = ("chain", "star", "binary")[neuron] if kind == "mixed" else kind
            parent = local - 1 if shape == "chain" else 0 if shape == "star" else (local - 1) // 2
            parents[base + local] = base + parent
    return parents


def _dense_reference(parents, coupling, diagonal, rhs):
    """Symmetric passive-cable matrix: diagonal minus child couplings, off-diagonal = coupling."""
    size = len(parents)
    compartments = size // 3
    expected = np.zeros((3, compartments), dtype=np.float64)
    for neuron in range(3):
        base = neuron * compartments
        matrix = np.diag(diagonal[base:base + compartments].astype(np.float64)).copy()
        for local in range(1, compartments):
            parent = parents[base + local] - base
            matrix[local, parent] = matrix[parent, local] = coupling[base + local]
        expected[neuron] = np.linalg.solve(matrix, rhs[base:base + compartments].astype(np.float64))
    return expected


@pytest.mark.parametrize("kind", ["chain", "star", "binary", "mixed"])
@pytest.mark.parametrize("compartments", [1, 2, 9, 65])
def test_r_edd_against_dense_reference(r_edd, kind, compartments):
    rng = np.random.default_rng(200 + compartments)
    neurons = 3
    size = neurons * compartments
    parents = _build_forest(kind, compartments, rng)
    coupling = -rng.uniform(0.01, 2.0, size).astype(np.float32)
    coupling[parents < 0] = 0
    diagonal = rng.uniform(1.0, 3.0, size).astype(np.float32)
    rhs = rng.uniform(-2.0, 2.0, size).astype(np.float32)

    child_offsets = np.zeros(size + 1, dtype=np.int32)
    for i in range(size):
        if parents[i] >= 0:
            child_offsets[i + 1] += 1
    for i in range(1, size + 1):
        child_offsets[i] += child_offsets[i - 1]
    child_indices = np.empty(child_offsets[-1], dtype=np.int32)
    next_idx = child_offsets.copy()
    for i in range(size):
        if parents[i] >= 0:
            child_indices[next_idx[parents[i]]] = i
            next_idx[parents[i]] += 1

    voltages = np.zeros(size, dtype=np.float32)
    spikes = np.zeros(neurons, dtype=np.int32)
    above = np.zeros(neurons, dtype=np.bool_)
    thresholds = np.zeros(neurons, dtype=np.float32)
    # Refresh timestep coefficients and RHS, as matrix construction does. Check
    # threshold history survives external assignments to the recorded voltages.
    for step in range(4):
        diagonal_step = diagonal.copy()
        rhs_step = rhs.copy()
        for child, parent in enumerate(parents):
            if parent >= 0:
                diagonal_step[child] -= coupling[child]
                diagonal_step[parent] -= coupling[child]
        expected = _dense_reference(parents, coupling, diagonal_step, rhs_step)
        thresholds = (expected[:, 0] + (0.5 if step == 2 else -0.5)).astype(np.float32)
        assert r_edd.r_edd_solve(parents, size, compartments, diagonal_step, rhs_step, coupling,
                                 thresholds, voltages, spikes, above) == 0
        np.testing.assert_allclose(voltages.reshape(3, compartments), expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(spikes, np.full(neurons, int(step in (0, 3)), dtype=np.int32))
        np.testing.assert_array_equal(above, np.full(neurons, step != 2, dtype=np.bool_))

