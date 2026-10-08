# -*- coding: utf-8 -*-
#
# test__code_generator_options.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

import pytest

from pynestml.codegeneration.nest_compartmental_code_generator import NESTCompartmentalCodeGenerator


@pytest.mark.parametrize("precision", ["single", "double"])
def test_fp_precision_option_is_accepted(precision):
    code_generator = NESTCompartmentalCodeGenerator({"nest_version": "v3", "fp_precision": precision})

    assert code_generator.get_option("fp_precision") == precision


@pytest.mark.parametrize("precision", ["half", 32, None])
def test_invalid_fp_precision_option_is_rejected(precision):
    with pytest.raises(ValueError):
        NESTCompartmentalCodeGenerator({"nest_version": "v3", "fp_precision": precision})
