# -*- coding: utf-8 -*-
#
# test__gpu_fastexp.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#

"""Codegen checks for the GPU side of the ``use_fastexp`` option.

With ``use_fastexp`` enabled, the exponential in the analytic-solver
propagator init expressions of the mechanism kernels must be replaced by the
device-side polynomial approximation ``cm_fast_propagator_exp`` (declared in the
generated ``fastexp.h``). Without it, the propagators keep using ``std::exp``.
"""
from pathlib import Path

import pytest

from pynestml.codegeneration.nest_gpu_compartmental_code_generator import NESTGPUCompartmentalCodeGenerator
from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target

CHANNEL_CU = "cm_group_channel_currents_cm_default_nestml.cu"
FASTEXP_H = "fastexp.h"
MODEL_PATH = str(Path(__file__).parent / "resources" / "cm_default.nestml")


@pytest.fixture(scope="module", params=[True, False])
def generated_channel_module(request, tmp_path_factory):
    use_fastexp = request.param
    tmp_path = tmp_path_factory.mktemp("fastexp_" + str(use_fastexp))
    gpu = tmp_path / "nest_gpu"
    src = gpu / "src"
    src.mkdir(parents=True)
    (src / "CMakeLists.txt").write_text("# <<BEGIN_NESTML_GENERATED>>\n# <<END_NESTML_GENERATED>>\n")
    target = tmp_path / "generated"
    options = dict(nest_gpu_path=str(gpu), register_neuron_model=False, skip_build=True, use_fastexp=use_fastexp)
    generate_nest_gpu_compartmental_target(
        input_path=MODEL_PATH,
        target_path=str(target), module_name="cm_default_module", suffix="_nestml",
        logging_level="ERROR", dev=True, codegen_opts=options,
    )
    return use_fastexp, target


def test_propagator_init_uses_fastexp(generated_channel_module):
    use_fastexp, target = generated_channel_module
    source = (target / CHANNEL_CU).read_text()
    header = (target / FASTEXP_H).read_text()

    if use_fastexp:
        assert "#include \"fastexp.h\"" in source
        assert source.count("cm_fast_propagator_exp((-__h) / tau_m_Na") >= 1
        assert "cm_fast_propagator_exp((-__h) / tau_n_K" in source
        # The header provides the device-side polynomial approximation.
        assert "__device__" in header and "__forceinline__" in header
        assert header.count("cm_fast_propagator_exp") >= 2
    else:
        assert "#include \"fastexp.h\"" not in source
        assert "cm_fast_propagator_exp" not in source
        assert "std::exp((-__h) / tau_m_Na" in source


def test_fastexp_option_is_accepted():
    generator = NESTGPUCompartmentalCodeGenerator({"use_fastexp": True})
    assert generator.get_option("use_fastexp") is True
