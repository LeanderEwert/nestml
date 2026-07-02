# -*- coding: utf-8 -*-
#
# nest_gpu_compartmental_code_generator.py
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

from __future__ import annotations

import copy
import glob
import os
import shutil
from typing import Any, Dict, Mapping, Optional, Sequence

from pynestml.codegeneration.nest_gpu_code_generator_utils import NESTGPUCodeGeneratorUtils
from pynestml.codegeneration.nest_compartmental_code_generator import NESTCompartmentalCodeGenerator
from pynestml.frontend.frontend_configuration import FrontendConfiguration
from pynestml.meta_model.ast_model import ASTModel
from pynestml.utils.ast_vector_parameter_setter_and_printer_factory import ASTPreAndSuffixSetterAndPrinterFactory
from pynestml.utils.logger import Logger, LoggingLevel


class NESTGPUCompartmentalCodeGenerator(NESTCompartmentalCodeGenerator):
    r"""
    Experimental code generator for compartmental models with CUDA/GPU-oriented templates.

    The generator deliberately reuses the compartmental analysis pipeline from
    :class:`NESTCompartmentalCodeGenerator` and only specializes template
    selection, generated file naming, and GPU-specific namespace helpers.
    """

    _default_options = copy.deepcopy(NESTCompartmentalCodeGenerator._default_options)
    _default_options["templates"] = {
        "path": "resources_nest_gpu_compartmental/cm_neuron",
        "model_templates": {
            "neuron": [
                "cm_group_receptor_currents_@NEURON_NAME@.cu.jinja2",
                "cm_group_receptor_currents_@NEURON_NAME@.h.jinja2",
            ]
        },
        "module_templates": [],
    }
    _default_options["nest_gpu_path"] = None
    _default_options["register_neuron_model"] = False

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        super().__init__(None)
        if options:
            self.set_options(options)
        if not self.option_exists("nest_gpu_path") or not self.get_option("nest_gpu_path"):
            if "NEST_GPU" in os.environ:
                self.nest_gpu_path = os.environ["NEST_GPU"]
            else:
                self.nest_gpu_path = os.getcwd()
            self.set_options({"nest_gpu_path": self.nest_gpu_path})
            Logger.log_message(None, -1, "The NEST-GPU path was automatically detected as: " + self.nest_gpu_path, None,
                               LoggingLevel.INFO)
        else:
            self.nest_gpu_path = self.get_option("nest_gpu_path")

    def get_cm_syns_neuroncurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_receptor_currents_" + neuron.get_name()

    def _get_module_namespace(self, neurons: Sequence[ASTModel]) -> Dict:
        namespace = super()._get_module_namespace(neurons)
        namespace["is_gpu_compartmental"] = True
        return namespace

    def _get_neuron_model_namespace(self, neuron: ASTModel, paired_synapses: Optional[Sequence[ASTModel]] = None) -> Dict:
        namespace = super()._get_neuron_model_namespace(neuron, paired_synapses)
        namespace["is_gpu_compartmental"] = True
        namespace["cuda_printer"] = _CompartmentalCUDAPrinter(neuron, self._printer_no_origin)
        return namespace

    def generate_module_code(self, neurons: Sequence[ASTModel], metadata: Dict[str, Dict[str, Any]]) -> None:
        """
        Prepare the NEST-GPU source tree for recompilation.

        The current default template set only emits receptor/current support
        files, so model registration in neuron_models.h/.cu is opt-in until the
        target produces a complete BaseNeuron implementation.
        """
        self.copy_models_from_target_path()
        self.add_files_to_makefile()
        if self.get_option("register_neuron_model"):
            self.add_model_name_to_neuron_header(neurons)
            self.add_model_to_neuron_class(neurons)

    def copy_models_from_target_path(self):
        dst_path = os.path.join(self.nest_gpu_path, "src")
        for pattern in ["*.h", "*.cu"]:
            for file_path in glob.glob(os.path.join(FrontendConfiguration.get_target_path(), pattern)):
                shutil.copy(file_path, dst_path)

    def add_files_to_makefile(self):
        cmakelists_path = os.path.join(self.nest_gpu_path, "src", "CMakeLists.txt")
        shutil.copy(cmakelists_path, cmakelists_path + ".bak")

        generated_files = []
        for pattern in ["*.h", "*.cu"]:
            for file_path in sorted(glob.glob(os.path.join(FrontendConfiguration.get_target_path(), pattern))):
                generated_files.append("\n    " + os.path.basename(file_path))
        generated_files = "".join(generated_files) + "\n"
        NESTGPUCodeGeneratorUtils.replace_text_between_tags(cmakelists_path,
                                                            generated_files,
                                                            begin_tag="# <<BEGIN_NESTML_GENERATED>>",
                                                            end_tag="# <<END_NESTML_GENERATED>>")

    def add_model_name_to_neuron_header(self, neurons: Sequence[ASTModel]):
        neuron_models_h_path = os.path.join(self.nest_gpu_path, "src", "neuron_models.h")
        shutil.copy(neuron_models_h_path, neuron_models_h_path + ".bak")

        neuron_indexes = []
        neuron_names = []
        for neuron in neurons:
            neuron_indexes.append("\ni_" + neuron.get_name() + "_model,")
            neuron_names.append("\n, \"" + neuron.get_name() + "\"")

        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_h_path, "".join(neuron_indexes) + "\n")
        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_h_path, "".join(neuron_names) + "\n", rfind=True)

    def add_model_to_neuron_class(self, neurons: Sequence[ASTModel]):
        neuron_models_cu_path = os.path.join(self.nest_gpu_path, "src", "neuron_models.cu")
        shutil.copy(neuron_models_cu_path, neuron_models_cu_path + ".bak")

        include_files = []
        code_blocks = []
        for neuron in neurons:
            model_name = neuron.get_name()
            include_files.append("\n#include \"" + model_name + ".h\"")
            code_blocks.append("\n"
                               f"else if (model_name == neuron_model_name[i_{model_name}_model]) {{\n"
                               f"    n_ports = {len(neuron.get_spike_input_ports())};\n"
                               f"    {model_name} *{model_name}_group = new {model_name};\n"
                               f"    node_vect_.push_back({model_name}_group);\n"
                               " }")

        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_cu_path, "".join(include_files) + "\n")
        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_cu_path, "".join(code_blocks) + "\n", rfind=True)


class _CompartmentalCUDAPrinter:
    def __init__(self, neuron: ASTModel, printer):
        self.printer_factory = ASTPreAndSuffixSetterAndPrinterFactory(neuron, printer)

    def print(self, expression, index: str = "i", array_name: str = "y", black_list=None):
        black_list = black_list or []
        index_printer = self.printer_factory.create_ast_pre_and_suffix_setter_and_printer(
            prefix=array_name + "[i_",
            suffix="+" + index + "]",
            black_list=black_list,
        )
        return index_printer.print(expression)

    def printer(self, index: str = "i", black_list=None):
        black_list = black_list or []
        return self.printer_factory.create_ast_vector_parameter_setter_and_printer(index, black_list)
