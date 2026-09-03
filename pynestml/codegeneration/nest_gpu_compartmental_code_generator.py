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
                "@NEURON_NAME@.cu.jinja2",
                "@NEURON_NAME@.h.jinja2",
                "cm_group_receptor_currents_@NEURON_NAME@.cu.jinja2",
                "cm_group_receptor_currents_@NEURON_NAME@.h.jinja2",
                "cm_group_channel_currents_@NEURON_NAME@.cu.jinja2",
                "cm_group_channel_currents_@NEURON_NAME@.h.jinja2",
                "cm_group_concentration_currents_@NEURON_NAME@.cu.jinja2",
                "cm_group_concentration_currents_@NEURON_NAME@.h.jinja2",
                "cm_group_continuous_input_currents_@NEURON_NAME@.cu.jinja2",
                "cm_group_continuous_input_currents_@NEURON_NAME@.h.jinja2",
                "cm_group_currents_@NEURON_NAME@.h.jinja2",
                "cm_tree_@NEURON_NAME@.cu.jinja2",
                "cm_tree_@NEURON_NAME@.h.jinja2",
            ]
        },
        "module_templates": [],
    }
    _default_options["nest_gpu_path"] = None
    _default_options["register_neuron_model"] = True
    _default_options["skip_build"] = False

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

    def set_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        ret = super().set_options(options)
        if options and "nest_gpu_path" in options:
            self.nest_gpu_path = self.get_option("nest_gpu_path")
        return ret

    def get_cm_syns_neuroncurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_currents_" + neuron.get_name()

    def get_cm_syns_receptorcurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_receptor_currents_" + neuron.get_name()

    def get_cm_syns_channelcurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_channel_currents_" + neuron.get_name()

    def get_cm_syns_concentrationcurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_concentration_currents_" + neuron.get_name()

    def get_cm_syns_continuouscurrents_file_prefix(self, neuron: ASTModel):
        return "cm_group_continuous_input_currents_" + neuron.get_name()

    def _get_module_namespace(self, neurons: Sequence[ASTModel]) -> Dict:
        namespace = super()._get_module_namespace(neurons)
        namespace["is_gpu_compartmental"] = True
        for neuron in neurons:
            namespace["perNeuronFileNamesCm"][neuron.get_name()].update({
                "groupcurrents": self.get_cm_syns_neuroncurrents_file_prefix(neuron),
                "receptorcurrents": self.get_cm_syns_receptorcurrents_file_prefix(neuron),
                "channelcurrents": self.get_cm_syns_channelcurrents_file_prefix(neuron),
                "concentrationcurrents": self.get_cm_syns_concentrationcurrents_file_prefix(neuron),
                "continuouscurrents": self.get_cm_syns_continuouscurrents_file_prefix(neuron),
            })
        return namespace

    def _get_neuron_model_namespace(self, neuron: ASTModel, paired_synapses: Optional[Sequence[ASTModel]] = None) -> Dict:
        namespace = super()._get_neuron_model_namespace(neuron, paired_synapses)
        namespace["is_gpu_compartmental"] = True
        namespace["neuronSpecificFileNamesCmSyns"].update({
            "groupcurrents": self.get_cm_syns_neuroncurrents_file_prefix(neuron),
            "receptorcurrents": self.get_cm_syns_receptorcurrents_file_prefix(neuron),
            "channelcurrents": self.get_cm_syns_channelcurrents_file_prefix(neuron),
            "concentrationcurrents": self.get_cm_syns_concentrationcurrents_file_prefix(neuron),
            "continuouscurrents": self.get_cm_syns_continuouscurrents_file_prefix(neuron),
        })
        namespace["cuda_printer"] = _CompartmentalCUDAPrinter(neuron, self._printer_no_origin)
        self._enrich_mechanism_usage_flags(namespace)
        return namespace

    @classmethod
    def _enrich_mechanism_usage_flags(cls, namespace: Dict[str, Any]) -> None:
        chan_info = namespace["chan_info"]
        recs_info = namespace["recs_info"]
        conc_info = namespace["conc_info"]
        con_in_info = namespace["con_in_info"]
        syns_info = namespace["syns_info"]

        for info in chan_info.values():
            info["requires_shared_current"] = False
            info["has_self_spike_processing"] = cls._has_non_empty_block(info, "SelfSpikesFunction")
        for info in recs_info.values():
            info["requires_shared_current"] = False
            info["has_self_spike_processing"] = (
                cls._has_non_empty_block(info, "SelfSpikesFunction")
                or cls._has_self_spike_convolution(info)
            )
        for info in conc_info.values():
            info["requires_shared_concentration"] = False
            info["has_self_spike_processing"] = cls._has_non_empty_block(info, "SelfSpikesFunction")
        for info in con_in_info.values():
            info["requires_shared_current"] = False
            info["has_self_spike_processing"] = cls._has_non_empty_block(info, "SelfSpikesFunction")

        current_infos = list(chan_info.values()) + list(recs_info.values()) + list(con_in_info.values())
        mechanism_infos = current_infos + list(conc_info.values())

        for consumer_info in mechanism_infos + list(syns_info.values()):
            cls._mark_required_shared_outputs(consumer_info, chan_info, recs_info, conc_info, con_in_info)

        namespace["has_mechanism_dependencies"] = any(
            info.get("requires_shared_current", False)
            for info in current_infos
        ) or any(info.get("requires_shared_concentration", False) for info in conc_info.values())

        namespace["has_self_spike_processing"] = any(
            info.get("has_self_spike_processing", False)
            for info in mechanism_infos
        )

    @staticmethod
    def _mark_required_shared_outputs(consumer_info: Dict[str, Any],
                                      chan_info: Dict[str, Dict[str, Any]],
                                      recs_info: Dict[str, Dict[str, Any]],
                                      conc_info: Dict[str, Dict[str, Any]],
                                      con_in_info: Dict[str, Dict[str, Any]]) -> None:
        dependencies = consumer_info.get("Dependencies") or {}
        for dependency in dependencies.get("channels", []):
            dependency_name = getattr(dependency, "variable_name", None)
            if dependency_name in chan_info:
                chan_info[dependency_name]["requires_shared_current"] = True

        for dependency in dependencies.get("receptors", []):
            dependency_name = getattr(dependency, "variable_name", None)
            if dependency_name in recs_info:
                recs_info[dependency_name]["requires_shared_current"] = True

        for dependency in dependencies.get("continuous", []):
            dependency_name = getattr(dependency, "variable_name", None)
            if dependency_name in con_in_info:
                con_in_info[dependency_name]["requires_shared_current"] = True

        for dependency in dependencies.get("concentrations", []):
            lhs = getattr(dependency, "lhs", None)
            dependency_name = getattr(lhs, "name", None)
            if dependency_name in conc_info:
                conc_info[dependency_name]["requires_shared_concentration"] = True

    @staticmethod
    def _has_non_empty_block(info: Dict[str, Any], block_name: str) -> bool:
        blocks = info.get("Blocks")
        block = blocks.get(block_name) if blocks and block_name in blocks else None
        if block is None:
            return False
        stmts_body = block.get_stmts_body()
        return bool(stmts_body and stmts_body.get_stmts())

    @staticmethod
    def _has_self_spike_convolution(info: Dict[str, Any]) -> bool:
        for convolution_info in info.get("convolutions", {}).values():
            spikes_info = convolution_info.get("spikes") or {}
            if spikes_info.get("name") == "self_spikes":
                return True
        return False

    def generate_module_code(self, neurons: Sequence[ASTModel], metadata: Dict[str, Dict[str, Any]]) -> None:
        """
        Prepare the NEST-GPU source tree for recompilation.

        The generated BaseNeuron wrapper references the NEST-GPU model enum, so
        registration in neuron_models.h/.cu is enabled by default for real
        builds. Tests can disable it when using a fake NEST-GPU source tree.
        """
        self.copy_models_from_target_path()
        self.add_files_to_makefile()
        if self.get_option("register_neuron_model"):
            self.add_model_name_to_neuron_header(neurons)
            self.add_model_to_neuron_class(neurons)

    def copy_models_from_target_path(self):
        dst_path = os.path.join(self.nest_gpu_path, "src")
        for pattern in ["*.h", "*.cu", "*.cpp"]:
            for file_path in glob.glob(os.path.join(FrontendConfiguration.get_target_path(), pattern)):
                shutil.copy(file_path, dst_path)

    def add_files_to_makefile(self):
        cmakelists_path = os.path.join(self.nest_gpu_path, "src", "CMakeLists.txt")
        shutil.copy(cmakelists_path, cmakelists_path + ".bak")

        generated_files = []
        for pattern in ["*.h", "*.cu", "*.cpp"]:
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
        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_h_path, "".join(neuron_names) + ",\n", rfind=True)

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
                               "    n_ports = 0;\n"
                               f"    {model_name} *{model_name}_group = new {model_name};\n"
                               f"    node_vect_.push_back({model_name}_group);\n"
                               " }")

        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_cu_path, "".join(include_files) + "\n")
        NESTGPUCodeGeneratorUtils.replace_text_between_tags(neuron_models_cu_path, "".join(code_blocks) + "\n", rfind=True)


class _CompartmentalCUDAPrinter:
    def __init__(self, neuron: ASTModel, printer):
        self.printer_factory = ASTPreAndSuffixSetterAndPrinterFactory(neuron, printer)
        self.parameter_names = {
            symbol.get_symbol_name()
            for symbol in list(neuron.get_parameter_symbols()) + list(neuron.get_internal_symbols())
            if symbol.get_symbol_name() != "__h" and not symbol.get_symbol_name().startswith("__P__")
        }

    def print(self, expression, index: str = "i", array_name: str = "y", black_list=None, stride: Optional[str] = None,
              param_index: Optional[str] = None, param_stride: Optional[str] = None):
        black_list = black_list or []
        suffix = "*" + stride + "+" + index + "]" if stride else "+" + index + "]"
        index_printer = self.printer_factory.create_ast_pre_and_suffix_setter_and_printer(
            prefix=array_name + "[i_",
            suffix=suffix,
            black_list=black_list,
        )
        code = index_printer.print(expression)
        if array_name != "param":
            param_index = param_index or index
            param_stride = param_stride if param_stride is not None else stride
            param_suffix = "*" + param_stride + "+" + param_index + "]" if param_stride else "+" + param_index + "]"
            for parameter_name in self.parameter_names:
                code = code.replace(
                    array_name + "[i_" + parameter_name + suffix,
                    "param[i_" + parameter_name + param_suffix)
        for variable_name in black_list:
            code = code.replace(array_name + "[i_" + variable_name + suffix, variable_name)
        return code

    def printer(self, index: str = "i", black_list=None, stride: Optional[str] = None,
                param_index: Optional[str] = None, param_stride: Optional[str] = None):
        black_list = black_list or []
        return _CompartmentalCUDABlockPrinter(self, index, black_list, stride, param_index, param_stride)


class _CompartmentalCUDABlockPrinter:
    def __init__(self, printer: _CompartmentalCUDAPrinter, index: str = "i", black_list=None,
                 stride: Optional[str] = None, param_index: Optional[str] = None, param_stride: Optional[str] = None):
        self.printer = printer
        self.index = index
        self.black_list = black_list or []
        self.stride = stride
        self.param_index = param_index
        self.param_stride = param_stride
        self.std_vector_parameter = index

    def print(self, node):
        return self.printer.print(node, self.index, black_list=self.black_list, stride=self.stride,
                                  param_index=self.param_index, param_stride=self.param_stride)
