# -*- coding: utf-8 -*-
"""One-shot validation: build hines into a temp target and run one chain case."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TARGET = tempfile.mkdtemp(prefix="val_hines_")
print(f"BUILD TARGET: {TARGET}")

from pynestml.frontend.pynestml_frontend import generate_nest_gpu_compartmental_target
import shutil

input_path = os.path.join(os.path.dirname(__file__), "resources", "cm_default.nestml")
if os.path.isdir(TARGET):
    shutil.rmtree(TARGET)
generate_nest_gpu_compartmental_target(
    input_path=input_path, target_path=TARGET, module_name="cm_default_gpu_module",
    suffix="_nestml", logging_level="ERROR", dev=True,
    codegen_opts={"register_neuron_model": True, "skip_build": False,
                  "gpu_tree_solver": "hines", "gpu_tree_solver_chain_length": 3, "gpu_tree_solver_base_size": 1},
)
with open(os.path.join(TARGET, "built.marker"), "w") as f:
    f.write("hines")
print("BUILD DONE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests", "nest_gpu_compartmental_tests"))
import gpu_compartmental_solver_runner as runner
out = tempfile.mkdtemp(prefix="val_run_")
config = runner.build_neuron_configuration("chain", 8, runner.OFF_EQUILIBRIUM_VOLTAGE)
result = runner.run_solver_simulation(config, sample_compartments=[0, 1, 8], dt=0.1, sim_time=1000.0)
print(json.dumps(result, indent=2))
with open(os.path.join(out, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print("RUN DONE")
