# -*- coding: utf-8 -*-
#
# compartmental_stdp_test.py
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

import os
import unittest

import numpy as np
import pytest

import nest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target

# set to `True` to plot simulation traces
TEST_PLOTS = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
except BaseException as e:
    # always set TEST_PLOTS to False if matplotlib can not be imported
    TEST_PLOTS = False

class TestCompartmentalConcmech(unittest.TestCase):
    @pytest.fixture(scope="module", autouse=True)
    def setup(self):
        tests_path = os.path.realpath(os.path.dirname(__file__))
        neuron_input_path = os.path.join(
            tests_path,
            "resources",
            "continuous_test.nestml"
        )
        synapse_input_path = os.path.join(
            tests_path,
            "resources",
            "stdp_synapse.nestml"
        )
        target_path = os.path.join(
            tests_path,
            "target/"
        )

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        print(
            f"Compiled nestml model 'cm_main_cm_default_nestml' not found, installing in:"
            f"    {target_path}"
        )

        nest.ResetKernel()
        nest.SetKernelStatus(dict(resolution=.1))
        if True:
            generate_nest_compartmental_target(
                input_path=[neuron_input_path, synapse_input_path],
                target_path=target_path,
                module_name="cm_stdp_module",
                suffix="_nestml",
                logging_level="INFO",
                codegen_opts={"neuron_synapse_pairs": [{"neuron": "continuous_test_model",
                                                        "synapse": "stdp_synapse",
                                                        "post_ports": ["post_spikes"]}],
                              "delay_variable": {"stdp_synapse": "d"},
                              "weight_variable": {"stdp_synapse": "w"}
                              }
            )

        nest.Install("cm_stdp_module.so")

    def test_cm_stdp(self):
        pre_spike_times = [1, 200]
        post_spike_times = [2, 199]
        sim_time = max(np.amax(pre_spike_times), np.amax(post_spike_times)) + 5
        wr = nest.Create("weight_recorder")
        nest.CopyModel("stdp_synapse_nestml__with_continuous_test_model_nestml", "stdp_nestml_rec",
                       {"weight_recorder": wr[0], "w": 1., "d": 1., "receptor_type": 0})
        #nest.CopyModel("stdp_synapse", "stdp_nestml_rec",
        #               {"weight_recorder": wr[0], "receptor_type": 0})
        external_input_pre = nest.Create("spike_generator", params={"spike_times": pre_spike_times})
        external_input_post = nest.Create("spike_generator", params={"spike_times": post_spike_times})
        pre_neuron = nest.Create("parrot_neuron")
        post_neuron = nest.Create('continuous_test_model_nestml__with_stdp_synapse_nestml')

