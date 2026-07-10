# -*- coding: utf-8 -*-
#
# test__gpu_compartmental_codegen.py
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

from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target


class TestNESTGPUCompartmentalCodeGenerator:
    TARGET_DIR = "target"
    MODEL_NAME = "cm_ampa_only_nestml"

    generated_files = {
        "cm_ampa_only_nestml.cu",
        "cm_ampa_only_nestml.h",
        "cm_group_currents_cm_ampa_only_nestml.h",
        "cm_group_receptor_currents_cm_ampa_only_nestml.cu",
        "cm_group_receptor_currents_cm_ampa_only_nestml.h",
        "cm_tree_cm_ampa_only_nestml.cpp",
        "cm_tree_cm_ampa_only_nestml.h",
    }

    def generate_receptor_only_model(self, target_path, nest_gpu_path=None, register_neuron_model=True, skip_build=False):
        tests_path = os.path.realpath(os.path.dirname(__file__))
        input_path = os.path.join(tests_path, "resources", "cm_ampa_only.nestml")
        codegen_opts = {
            "register_neuron_model": register_neuron_model,
            "skip_build": skip_build,
        }
        if nest_gpu_path is not None:
            codegen_opts["nest_gpu_path"] = str(nest_gpu_path)

        generate_nest_gpu_compartmental_target(
            input_path=input_path,
            target_path=str(target_path),
            module_name="cm_ampa_only_gpu_module",
            suffix="_nestml",
            logging_level="ERROR",
            dev=True,
            codegen_opts=codegen_opts,
        )

    def test_receptor_only_generation_into_real_nest_gpu_source(self):
        tests_path = os.path.realpath(os.path.dirname(__file__))
        target_path = os.path.join(tests_path, self.TARGET_DIR)
        nest_gpu_path = os.environ.get("NEST_GPU", os.getcwd())
        nest_gpu_src_path = os.path.join(nest_gpu_path, "src")
        cmakelists_path = os.path.join(nest_gpu_src_path, "CMakeLists.txt")

        assert os.path.isdir(nest_gpu_src_path)
        assert os.path.isfile(cmakelists_path)

        self.generate_receptor_only_model(target_path)

        for filename in self.generated_files:
            assert os.path.isfile(os.path.join(target_path, filename))
            assert os.path.isfile(os.path.join(nest_gpu_src_path, filename))

        with open(cmakelists_path, encoding="utf-8") as cmakelists_file:
            cmakelists = cmakelists_file.read()
        for filename in self.generated_files:
            assert filename in cmakelists

        self.run_receptor_only_simulation_smoke()

    def run_receptor_only_simulation_smoke(self):
        import nestgpu as ngpu

        neuron = ngpu.Create(self.MODEL_NAME, 1)

        ngpu.SetStatus(neuron, {
            "compartments": [{"parent_idx": -1, "params": {}}],
            "receptors": [{"comp_idx": 0, "receptor_type": "AMPA"}],
        })

        spike_generator = ngpu.Create("spike_generator")
        ngpu.SetStatus(spike_generator, {"spike_times": list(range(1, 5))})

        conn_dict = {"rule": "one_to_one"}
        syn_dict = {"weight": 1.0, "delay": 2.0, "receptor": 0}

        ngpu.Connect(spike_generator, neuron, conn_dict, syn_dict)
        ngpu.Simulate(6.0)
