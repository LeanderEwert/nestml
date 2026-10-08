# -*- coding: utf-8 -*-
#
# conftest.py
#
# This file is part of NEST.
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

import os

import pytest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_compartmental_target


CPU_SINGLE_PRECISION_MODEL = "cm_default_cpu_single_nestml"
CPU_SINGLE_PRECISION_MODULE = "cm_default_cpu_single_module"
SKIP_REBUILD_ENV = "NESTML_GPU_CM_SKIP_REBUILD"


@pytest.fixture(scope="session")
def cpu_single_precision_cm_default():
    tests_path = os.path.realpath(os.path.dirname(__file__))
    input_path = os.path.join(
        tests_path, "..", "nest_compartmental_tests", "resources", "cm_default.nestml")
    target_path = os.path.join(tests_path, "target_cpu_single_precision")
    install_path = os.path.join(target_path, "install")
    module_path = os.path.join(install_path, f"{CPU_SINGLE_PRECISION_MODULE}.so")

    if os.environ.get(SKIP_REBUILD_ENV) in ("1", "true", "True", "yes", "YES"):
        assert os.path.isfile(module_path), (
            f"{SKIP_REBUILD_ENV} is set, but the CPU reference module does not exist at {module_path}")
        print(f"Skipping single-precision CPU cm_default rebuild because {SKIP_REBUILD_ENV} is set")
    else:
        os.makedirs(install_path, exist_ok=True)
        generate_nest_compartmental_target(
            input_path=input_path,
            target_path=target_path,
            install_path=install_path,
            module_name=CPU_SINGLE_PRECISION_MODULE,
            suffix="_cpu_single_nestml",
            logging_level="ERROR",
            codegen_opts={
                "fp_precision": "single",
                "nest_version": NESTTools.detect_nest_version(),
            },
        )

    return {
        "model_name": CPU_SINGLE_PRECISION_MODEL,
        "module_path": module_path,
    }
