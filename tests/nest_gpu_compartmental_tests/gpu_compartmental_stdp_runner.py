"""Run one compartmental STDP timing pair in a fresh NEST-GPU process."""
import argparse
import json


DT = .1
POST_TIME = 10.
MEASURING_TIME = 21.
END_TIME = 22.
INITIAL_WEIGHT = 10.
STDP_PARAMS = {"lambda": .01, "alpha": 1., "mu_plus": 1., "mu_minus": 1., "Wmax": 100.}


def configure(neuron, generated, receptor_delay=DT):
    import nestgpu as ngpu

    params = {"e_AMPA": -70.}
    if generated:
        params.update(STDP_PARAMS, w=INITIAL_WEIGHT, delay=receptor_delay,
                      tau_tr_pre=20., tau_tr_post=20., Wmin=0.)
    ngpu.SetStatus(neuron, {
        "V_th": -55.,
        "compartments": [{"parent_idx": -1, "params": {
            "C_m": 10., "g_C": 0., "g_L": 1.5, "e_L": -70., "v_comp": -70.}}],
        "receptors": [{"comp_idx": 0,
                       "receptor_type": "AMPA_stdp_synapse_nestml" if generated else "AMPA",
                       "params": params}],
    })


def run_pair(model, pre_time, receptor_delay=DT):
    import nestgpu as ngpu

    ngpu.SetTimeResolution(DT)
    ngpu.SetKernelStatus({"spike_buffer_algo": 0})
    generator = ngpu.Create("spike_generator")
    pre = ngpu.Create("parrot_neuron")
    ngpu.SetStatus(generator, {"spike_times": [pre_time, MEASURING_TIME]})
    ngpu.Connect(generator, pre, {"rule": "one_to_one"}, {"weight": 2., "delay": DT})
    ngpu.ActivateRecSpikeTimes(pre, 100)

    native = ngpu.CreateSynGroup("stdp", {**STDP_PARAMS, "tau_plus": 20., "tau_minus": 20.})
    branches = {}
    for name, generated in (("native", False), ("generated", True)):
        neuron = ngpu.Create(model)
        configure(neuron, generated, receptor_delay)
        fields = ["v_comp0", "AMPA_stdp_synapse_nestml0" if generated else "i_tot_AMPA0"]
        if generated:
            fields += ["w0", "pre_trace0", "post_trace0"]
        record = ngpu.CreateRecord("", fields, [neuron[0]] * len(fields), [0] * len(fields))
        ngpu.ActivateRecSpikeTimes(neuron, 100)
        syn_spec = {"receptor": 0, "delay": DT, "weight": 1. if generated else INITIAL_WEIGHT}
        if not generated:
            syn_spec["synapse_group"] = native
        ngpu.Connect(pre, neuron, {"rule": "one_to_one"}, syn_spec)
        branches[name] = (neuron, record, fields)

    ngpu.Simulate(POST_TIME)
    for neuron, _, _ in branches.values():
        # Use the scalar setter: the compartmental dictionary setter configures
        # morphology/mechanisms, while this writes the live device state.
        ngpu.SetStatus(neuron, "v_comp0", 0.)
    ngpu.Simulate(END_TIME - POST_TIME)

    result = {"pre_time": pre_time, "offset": pre_time - POST_TIME,
              "pre_spikes": ngpu.GetRecSpikeTimes(pre)[0]}
    for name, (neuron, record, fields) in branches.items():
        rows = ngpu.GetRecordData(record)
        events = {field: [row[column] for row in rows]
                  for column, field in enumerate(["times"] + fields)}
        weight = (ngpu.GetStatus(neuron, "w0")[0][0] if name == "generated" else
                  ngpu.GetStatus(ngpu.GetConnections(pre, neuron), "weight")[0])
        result[name] = {"weight": weight, "post_spikes": ngpu.GetRecSpikeTimes(neuron)[0],
                        "events": events}
    return result


def run_threshold_memory(model):
    """External assignments must trigger once, then rearm after falling below."""
    import nestgpu as ngpu

    ngpu.SetTimeResolution(DT)
    ngpu.SetKernelStatus({"spike_buffer_algo": 0})
    neuron = ngpu.Create(model)
    configure(neuron, False)
    ngpu.ActivateRecSpikeTimes(neuron, 100)
    ngpu.Simulate(1.)
    assert not ngpu.GetRecSpikeTimes(neuron)[0]
    ngpu.SetStatus(neuron, "v_comp0", 0.)
    ngpu.Simulate(1.)
    snapshots = [ngpu.GetRecSpikeTimes(neuron)[0]]
    ngpu.SetStatus(neuron, "v_comp0", 0.)  # still above; must not spike again
    ngpu.Simulate(1.)
    snapshots.append(ngpu.GetRecSpikeTimes(neuron)[0])
    ngpu.SetStatus(neuron, "v_comp0", -70.)
    ngpu.Simulate(1.)
    snapshots.append(ngpu.GetRecSpikeTimes(neuron)[0])
    ngpu.SetStatus(neuron, "v_comp0", 0.)
    ngpu.Simulate(1.)
    snapshots.append(ngpu.GetRecSpikeTimes(neuron)[0])
    return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--pre-time", type=float)
    parser.add_argument("--receptor-delay", type=float, default=DT)
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    if not args.threshold_only and args.pre_time is None:
        parser.error("--pre-time is required for the STDP protocol")
    result = (run_threshold_memory(args.model) if args.threshold_only else
              run_pair(args.model, args.pre_time, args.receptor_delay))
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output)
