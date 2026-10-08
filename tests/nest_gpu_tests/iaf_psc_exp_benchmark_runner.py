# -*- coding: utf-8 -*-
#
# This file is part of NEST, distributed under the GNU General Public License
# version 2 or later. See the file LICENSE for details.
"""Isolated CPU/GPU worker for the built-in iaf_psc_exp benchmark.

Imports and device/kernel initialization are outside the measured intervals.
The first simulation includes calibration; the second uses the warmed simulator.
"""

import argparse
import ctypes
import json
from pathlib import Path
import time


PARAMETERS = {
    "C_m": 250.0, "tau_m": 10.0, "E_L": -70.0,
    "V_m": -70.0, "V_th": -55.0, "V_reset": -70.0,
    "t_ref": 2.0, "tau_syn_ex": 2.0, "tau_syn_in": 5.0, "I_e": 500.0,
}
PSC_SPIKE_TIMES = [10.0, 30.0, 60.0]
PSC_DELAY = 1.0
VALIDATION_DURATION = 100.0


def gpu_parameters(parameters):
    """Translate absolute NEST voltages and time-constant names explicitly."""
    result = {key: parameters[key] for key in ("C_m", "tau_m", "E_L", "t_ref", "I_e")}
    result.update({
        "V_m_rel": parameters["V_m"] - parameters["E_L"],
        "Theta_rel": parameters["V_th"] - parameters["E_L"],
        "V_reset_rel": parameters["V_reset"] - parameters["E_L"],
        "tau_ex": parameters["tau_syn_ex"], "tau_in": parameters["tau_syn_in"],
        "I_syn_ex": 0.0, "I_syn_in": 0.0, "den_delay": 0.0,
    })
    return result


def cuda_synchronizer(ngpu):
    # Resolve CUDA through the library NEST GPU actually loaded, avoiding a
    # second, potentially different CUDA runtime. Fail rather than time launches.
    library = ctypes.CDLL(ngpu.lib_path)
    synchronize = library.cudaDeviceSynchronize
    synchronize.argtypes = []
    synchronize.restype = ctypes.c_int
    error_string = library.cudaGetErrorString
    error_string.argtypes = [ctypes.c_int]
    error_string.restype = ctypes.c_char_p

    def sync():
        status = synchronize()
        if status:
            raise RuntimeError(f"CUDA synchronization failed: {error_string(status).decode()}")

    return sync


def run_cpu(args):
    import nest

    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    # Otherwise adding a multimeter changes NEST's default communication
    # interval, confounding the comparison between recording modes.
    nest.SetKernelStatus({"resolution": args.dt, "local_num_threads": args.threads,
                          "min_delay": PSC_DELAY, "max_delay": PSC_DELAY})
    metadata = {"version": nest.__version__, "module_path": nest.__file__,
                "threads": args.threads, "min_delay_ms": PSC_DELAY, "max_delay_ms": PSC_DELAY,
                "precision": "built-in CPU model (double)"}
    started = time.perf_counter()
    count = 3 if args.validation else args.neurons
    neurons = nest.Create("iaf_psc_exp", count, PARAMETERS)
    if args.validation:
        neurons[1:].I_e = 0.0
        for index, weight in ((1, 200.0), (2, -200.0)):
            source = nest.Create("spike_generator", params={"spike_times": PSC_SPIKE_TIMES})
            nest.Connect(source, neurons[index], syn_spec={"weight": weight, "delay": PSC_DELAY})
        spikes = nest.Create("spike_recorder")
        nest.Connect(neurons, spikes)
    recorders = []
    if args.record or args.validation:
        indices = range(count) if args.validation else [args.sample]
        variables = ["V_m", "I_syn_ex", "I_syn_in"] if args.validation else ["V_m"]
        for index in indices:
            meter = nest.Create("multimeter", params={"record_from": variables, "interval": args.dt})
            nest.Connect(meter, neurons[index])
            recorders.append((index, meter))
    setup_s = time.perf_counter() - started
    duration = VALIDATION_DURATION if args.validation else args.duration
    first_started = time.perf_counter()
    nest.Simulate(duration)
    first_finished = time.perf_counter()
    warm_s = None
    if not args.validation:
        warm_started = time.perf_counter()
        nest.Simulate(duration)
        warm_s = time.perf_counter() - warm_started
    retrieval_started = time.perf_counter()
    traces = []
    for index, meter in recorders:
        events = nest.GetStatus(meter, "events")[0]
        traces.append({"neuron": index, **{key: events[key].tolist() for key in ["times", *variables]}})
    spike_times = None
    if args.validation:
        events = nest.GetStatus(spikes, "events")[0]
        spike_times = [events["times"][events["senders"] == node].tolist() for node in neurons.tolist()]
    retrieval_s = time.perf_counter() - retrieval_started
    return {"metadata": metadata, "setup_s": setup_s,
            "first_simulation_s": first_finished - first_started,
            "setup_first_s": first_finished - started, "warm_simulation_s": warm_s,
            "retrieval_s": retrieval_s, "traces": traces, "spike_times": spike_times}


def run_gpu(args):
    import nestgpu as ngpu

    ngpu.SetVerbosityLevel(0)
    ngpu.SetTimeResolution(args.dt)
    # Keep the engine configuration fixed throughout the sweep.
    ngpu.SetKernelStatus({"spike_buffer_algo": args.gpu_spike_buffer_algo})
    sync = cuda_synchronizer(ngpu)
    sync()
    library_path = Path(ngpu.lib_path).resolve()
    library_stat = library_path.stat()
    library = ctypes.CDLL(ngpu.lib_path)
    device = ctypes.c_int()
    library.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.cudaGetDevice.restype = ctypes.c_int
    if library.cudaGetDevice(ctypes.byref(device)):
        raise RuntimeError("Cannot identify active CUDA device")
    pci_bus_id = ctypes.create_string_buffer(64)
    library.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    library.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    if library.cudaDeviceGetPCIBusId(pci_bus_id, len(pci_bus_id), device):
        raise RuntimeError("Cannot identify active CUDA device PCI bus ID")
    metadata = {"version": getattr(ngpu, "__version__", "not exposed by Python API"),
                "module_path": ngpu.__file__, "library_path": str(library_path),
                "library_mtime_ns": library_stat.st_mtime_ns, "library_size": library_stat.st_size,
                "cuda_device": device.value, "pci_bus_id": pci_bus_id.value.decode(),
                "spike_buffer_algo": args.gpu_spike_buffer_algo,
                "precision": "built-in GPU model (float)",
                "synchronization": "cudaDeviceSynchronize through NEST GPU library"}
    started = time.perf_counter()
    count = 3 if args.validation else args.neurons
    neurons = ngpu.Create("iaf_psc_exp", count)
    ngpu.SetStatus(neurons, gpu_parameters(PARAMETERS))
    if args.validation:
        ngpu.SetStatus(ngpu.NodeSeq(neurons[1], 2), {"I_e": 0.0})
        for index, weight, receptor in ((1, 200.0, 0), (2, -200.0, 1)):
            source = ngpu.Create("spike_generator")
            # NEST GPU delivers generator spikes one update later than NEST.
            # Advance emission by one grid step to match their effect on V_m.
            ngpu.SetStatus(source, {"spike_times": [t - args.dt for t in PSC_SPIKE_TIMES]})
            ngpu.Connect(source, [neurons[index]], {"rule": "all_to_all"},
                         {"weight": weight, "delay": PSC_DELAY, "receptor": receptor})
        ngpu.ActivateRecSpikeTimes(neurons, 1000)
    recorders = []
    if args.record or args.validation:
        indices = range(count) if args.validation else [args.sample]
        variables = ["V_m_rel", "I_syn_ex", "I_syn_in"] if args.validation else ["V_m_rel"]
        for index in indices:
            meter = ngpu.CreateRecord("", variables, [neurons[index]] * len(variables), [0] * len(variables))
            recorders.append((index, meter))
    sync()
    setup_s = time.perf_counter() - started
    duration = VALIDATION_DURATION if args.validation else args.duration
    first_started = time.perf_counter()
    ngpu.Simulate(duration)
    sync()
    first_finished = time.perf_counter()
    warm_s = None
    if not args.validation:
        warm_started = time.perf_counter()
        ngpu.Simulate(duration)
        sync()
        warm_s = time.perf_counter() - warm_started
    retrieval_started = time.perf_counter()
    traces = []
    for index, meter in recorders:
        rows = ngpu.GetRecordData(meter)
        trace = {"neuron": index, "times": [row[0] for row in rows]}
        for column, variable in enumerate(variables, start=1):
            if variable == "V_m_rel":
                trace["V_m"] = [row[column] + PARAMETERS["E_L"] for row in rows]
            else:
                trace[variable] = [row[column] for row in rows]
        traces.append(trace)
    spike_times = ngpu.GetRecSpikeTimes(neurons) if args.validation else None
    retrieval_s = time.perf_counter() - retrieval_started
    return {"metadata": metadata, "setup_s": setup_s,
            "first_simulation_s": first_finished - first_started,
            "setup_first_s": first_finished - started, "warm_simulation_s": warm_s,
            "retrieval_s": retrieval_s, "traces": traces, "spike_times": spike_times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("cpu", "gpu"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--neurons", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpu-spike-buffer-algo", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    if args.neurons < 1 or not 0 <= args.sample < args.neurons or args.threads < 1:
        parser.error("neurons and threads must be positive; sample must be inside the population")
    if not args.dt > 0 or not args.duration >= args.dt:
        parser.error("require 0 < dt <= duration")
    result = run_cpu(args) if args.backend == "cpu" else run_gpu(args)
    result.update({"backend": args.backend, "n_neurons": 3 if args.validation else args.neurons,
                   "sample_neuron": args.sample, "recording_enabled": args.record or args.validation})
    args.output.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
