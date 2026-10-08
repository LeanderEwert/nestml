"""Exercise native spike recording, delivery, and STDP in a fresh GPU process."""
import json
import sys


def run(model):
    import nestgpu as ngpu

    ngpu.SetTimeResolution(.1)
    # Match NEST-GPU's built-in STDP tests.
    ngpu.SetKernelStatus({"spike_buffer_algo": 0})
    drive = ngpu.Create("spike_generator")
    pre = ngpu.Create("spike_generator")
    neurons = ngpu.Create(model, 4)
    ngpu.SetStatus(neurons, {
        "V_th": -50.,
        "compartments": [{"parent_idx": -1, "params": {
            "C_m": 10., "g_L": 1.5, "g_C": 0., "e_L": -70., "v_comp": -70.}}],
        "receptors": [
            {"comp_idx": 0, "receptor_type": "AMPA", "params": {"e_AMPA": 0.}},
            {"comp_idx": 0, "receptor_type": "AMPA", "params": {"e_AMPA": -70.}},
        ],
    })
    downstream = ngpu.Create("parrot_neuron", 3)
    for nodes in (neurons, downstream):
        ngpu.ActivateSpikeCount(nodes)
        ngpu.ActivateRecSpikeTimes(nodes, 100)
    ngpu.SetStatus(drive, {"spike_times": [15.]})
    ngpu.SetStatus(pre, {"spike_times": [5.]})
    ngpu.Connect(drive, [neurons[i] for i in (0, 2, 3)], {"rule": "all_to_all"},
                 {"receptor": 0, "weight": 50., "delay": .1})
    ngpu.Connect(neurons[0:3], downstream, {"rule": "one_to_one"}, {"weight": 1., "delay": .5})
    plastic = ngpu.CreateSynGroup("stdp", {"lambda": .01, "alpha": 1., "mu_plus": 1.,
                                         "mu_minus": 1., "tau_plus": 20., "tau_minus": 20., "Wmax": 100.})
    ngpu.Connect(pre, neurons, {"rule": "all_to_all"},
                 {"receptor": 1, "weight": 10., "delay": .1, "synapse_group": plastic})
    ngpu.Simulate(30.)
    weights = [ngpu.GetStatus(ngpu.GetConnections(pre, neurons[i:i + 1]), "weight")[0]
               for i in range(4)]
    return {"first_node": neurons[0], "spikes": ngpu.GetRecSpikeTimes(neurons),
            "counts": ngpu.GetStatus(neurons, "spike_count"),
            "downstream_spikes": ngpu.GetRecSpikeTimes(downstream), "weights": weights}


if __name__ == "__main__":
    result = run(sys.argv[1])
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        json.dump(result, output)
