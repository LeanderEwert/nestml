"""Run the same mixed ordinary/plastic receptor protocol in NEST or NEST GPU."""
import json
import sys

DT = .1
SIM_TIME = 40.
MODEL = "cm_ampa_only_nestml"
RECORDABLES = ["v_comp0", "w0", "w1", "w3", "pre_trace0", "post_trace0",
               "pre_trace1", "post_trace1", "pre_trace3", "post_trace3", "g_AMPA0", "g_AMPA1"]


def configuration(index):
    return {
        "V_th": -50.,
        "compartments": [{"parent_idx": -1, "params": {
            "C_m": 10., "g_L": 1.5, "g_C": 0., "e_L": -70., "v_comp": -70.}}],
        "receptors": [
            {"comp_idx": 0, "receptor_type": "AMPA_stdp_synapse_nestml",
             "params": {"w": 10. + index, "delay": DT, "e_AMPA": -70.}},
            {"comp_idx": 0, "receptor_type": "AMPA_stdp_nn_symm_synapse_nestml",
             "params": {"w": 20. + index, "delay": .4, "e_AMPA": -70.}},
            {"comp_idx": 0, "receptor_type": "AMPA", "params": {"e_AMPA": 0.}},
            {"comp_idx": 0, "receptor_type": "AMPA_third_factor_stdp_synapse_nestml",
             "params": {"w": 30., "d": DT, "e_AMPA": -70., "third_factor_scale": float(index)}},
        ],
    }


def run_gpu():
    import nestgpu as ngpu

    ngpu.SetTimeResolution(DT)
    neurons = ngpu.Create(MODEL, 2)
    records = []
    for index in range(2):
        ngpu.SetStatus(neurons[index:index + 1], configuration(index))
        records.append(ngpu.CreateRecord("", RECORDABLES, [neurons[index]] * len(RECORDABLES),
                                         [0] * len(RECORDABLES)))
    pre = ngpu.Create("spike_generator")
    drive = ngpu.Create("spike_generator")
    ngpu.SetStatus(pre, {"spike_times": [5., 25.]})
    ngpu.SetStatus(drive, {"spike_times": [15.]})
    for port in (0, 1, 3):
        ngpu.Connect(pre, neurons, {"rule": "all_to_all"}, {"receptor": port, "weight": 1., "delay": DT})
    ngpu.Connect(drive, neurons, {"rule": "all_to_all"}, {"receptor": 2, "weight": 50., "delay": DT})
    ngpu.Simulate(SIM_TIME)
    result = []
    for record in records:
        data = ngpu.GetRecordData(record)
        assert data and len(data[0]) == len(RECORDABLES) + 1
        result.append({name: [row[column] for row in data] for column, name in enumerate(["times"] + RECORDABLES)})
    return result


def run_cpu(module_path):
    import nest

    nest.ResetKernel()
    nest.Install(module_path)
    nest.SetKernelStatus({"resolution": DT})
    neurons = nest.Create(MODEL, 2)
    records = []
    for index in range(2):
        nest.SetStatus(neurons[index:index + 1], configuration(index))
        record = nest.Create("multimeter", params={"record_from": RECORDABLES, "interval": DT})
        nest.Connect(record, neurons[index:index + 1])
        records.append(record)
    # Same scheduler alignment used by test__gpu_compartmental_model.py.
    pre = nest.Create("spike_generator", params={"spike_times": [5. + 2*DT, 25. + 2*DT]})
    drive = nest.Create("spike_generator", params={"spike_times": [15. + 2*DT]})
    for port in (0, 1, 3):
        nest.Connect(pre, neurons, syn_spec={"receptor_type": port, "weight": 1., "delay": DT})
    nest.Connect(drive, neurons, syn_spec={"receptor_type": 2, "weight": 50., "delay": DT})
    nest.Simulate(SIM_TIME)
    return [{name: nest.GetStatus(record, "events")[0][name].tolist() for name in ["times"] + RECORDABLES}
            for record in records]


if __name__ == "__main__":
    results = run_gpu() if sys.argv[1] == "gpu" else run_cpu(sys.argv[3])
    with open(sys.argv[2], "w", encoding="utf-8") as output:
        json.dump(results, output)
