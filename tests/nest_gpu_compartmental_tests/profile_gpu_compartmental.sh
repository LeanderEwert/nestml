#!/usr/bin/env bash
# Profile the fixed-size GPU workloads without rebuilding the installed model.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: profile_gpu_compartmental.sh [--size N] [all|population|linear|star] [output_directory]

Run from an activated NESTML/NEST-GPU Python environment.
Each configuration gets its own Nsight Systems report and console log.
By default, a unique directory is created under report/profiling/.
--size sets the neuron count (population) or added dendritic compartment count
(linear/star). The soma is additional. Default: 4096. With all, N applies to all
three configurations. Example: profile_gpu_compartmental.sh linear --size 128

Environment:
  PYTHON                         Python executable (default: python3)
  NSYS                           Nsight Systems executable (default: nsys)
  QDSTRM_IMPORTER                 Optional path to the matching QdstrmImporter
  NSYS_CUDA_GRAPH_TRACE           graph (default) or node for kernel detail
  NESTML_GPU_CM_SKIP_REBUILD      Defaults to 1; set to 0 to rebuild in each run

Node-level tracing of the large linear tree can generate very large reports
and substantially slow the run. Let nsys finish exporting after pytest exits;
interrupting it may leave only an incomplete .qdstrm file.
EOF
}

find_qdstrm_importer() {
    local nsys_dir candidate
    if [[ -n "${QDSTRM_IMPORTER:-}" ]]; then
        command -v "$QDSTRM_IMPORTER"
        return
    fi
    nsys_dir=$(dirname -- "$(readlink -f -- "$(command -v "$nsys")")")
    for candidate in \
        "$nsys_dir/../host-linux-x64/QdstrmImporter" \
        "$nsys_dir/host-linux-x64/QdstrmImporter" \
        QdstrmImporter \
        /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return
        fi
    done
    return 1
}

size=4096
positional=()
while (( $# )); do
    case "$1" in
        --size)
            if [[ ! "${2:-}" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' '--size requires a positive integer.' >&2
                exit 2
            fi
            size=$2
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        --) shift; positional+=("$@"); break ;;
        --*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
        *) positional+=("$1"); shift ;;
    esac
done
set -- "${positional[@]}"
if (( $# > 2 )); then
    usage >&2
    exit 2
fi

case "${1:-all}" in
    all) configurations=(population linear star) ;;
    population|linear|star) configurations=("$1") ;;
    *) usage >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}
nsys=${NSYS:-nsys}
graph_trace=${NSYS_CUDA_GRAPH_TRACE:-graph}
for executable in "$python" "$nsys"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Executable not found: %s\n' "$executable" >&2
        exit 1
    fi
done
case "$graph_trace" in
    graph|node) ;;
    *) printf 'NSYS_CUDA_GRAPH_TRACE must be graph or node\n' >&2; exit 2 ;;
esac

if [[ -n "${2:-}" ]]; then
    mkdir -p -- "$2"
    output_dir=$(cd -- "$2" && pwd)
else
    mkdir -p -- "$script_dir/report/profiling"
    output_dir=$(mktemp -d "$script_dir/report/profiling/$(date +%Y%m%d_%H%M%S)_XXXXXX")
fi

export NESTML_GPU_CM_SKIP_REBUILD=${NESTML_GPU_CM_SKIP_REBUILD:-1}
export NESTML_GPU_CM_PROFILE_SIZE=$size
export PYTHONUNBUFFERED=1
cd -- "$script_dir"

printf 'Reports: %s\nCUDA graph tracing: %s\nWorkload size: %s\n' "$output_dir" "$graph_trace" "$size"
for configuration in "${configurations[@]}"; do
    case "$configuration" in
        population) test_name=test_population ;;
        linear) test_name=test_linear_compartments ;;
        star) test_name=test_star_compartments ;;
    esac
    report="$output_dir/cm_default_$configuration"
    if [[ -e "$report.nsys-rep" || -e "$report.qdstrm" || -e "$report.log" ]]; then
        printf 'Output already exists for %s; choose a new output directory.\n' "$report" >&2
        exit 1
    fi
    printf '\n=== Profiling %s; waiting for simulation and report export ===\n' "$configuration"
    "$nsys" profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --cuda-graph-trace="$graph_trace" \
        --wait=primary \
        --stop-on-exit=true \
        --show-output=true \
        --output="$report" \
        "$python" -m pytest -s -v \
        "test__gpu_compartmental_profiling.py::TestNESTGPUCompartmentalProfiling::$test_name" \
        2>&1 | tee "$report.log"
    if [[ ! -s "$report.nsys-rep" && -s "$report.qdstrm" ]]; then
        if ! importer=$(find_qdstrm_importer); then
            printf 'Set QDSTRM_IMPORTER to the QdstrmImporter from your Nsight version.\n' >&2
            exit 1
        fi
        printf 'Converting .qdstrm to .nsys-rep using %s; please wait...\n' "$importer" \
            | tee -a "$report.log"
        if ! "$importer" --input-file "$report.qdstrm" --output-file "$report.nsys-rep" \
            2>&1 | tee -a "$report.log"; then
            printf 'Conversion failed. Check %s.log and ensure QDSTRM_IMPORTER matches the Nsight version.\n' \
                "$report" >&2
            exit 1
        fi
    fi
    if [[ ! -s "$report.nsys-rep" ]]; then
        printf 'No .nsys-rep was produced. Check %s.log for export/import errors.\n' "$report" >&2
        exit 1
    fi
    printf 'Report ready: %s.nsys-rep\n' "$report"
done
printf '\nAll requested profiles finished. Open the .nsys-rep files in Nsight Systems.\n'
