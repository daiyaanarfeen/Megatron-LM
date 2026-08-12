#!/bin/bash

# Same-allocation full-original a3b/30b healthy EP16/EP16 versus NEP EP16/EP12.
# Iterations 1-4 warm the model, 5-7 are profiled, and 8-10 are clean timing samples.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b128e-ep16-12-postwarm-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_30b_original_128e_ep16_ep12_postwarm_ab}"
HEALTHY_WRAPPER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_128e_ep16_healthy_flex_no_optimizer.sh"
NEP_WRAPPER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_128e_ep16_ep12_flex_no_optimizer.sh"
ANALYZER="${REPO_DIR}/scripts/nonuniform/analyze_a3b_full_ep16_ep12_trace_ab.py"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
PROFILE_RANKS="${PROFILE_RANKS:-0 16}"
PROFILE_STEPS=$((PROFILE_STEP_END - PROFILE_STEP_START))

mkdir -p "${ROOT_DIR}"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 8)); then
    echo "Expected eight allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
FIRST_NODE="${allocated_nodes[0]}"

run_case() {
    local label="$1"
    local wrapper="$2"
    echo "[a3b-postwarm-ab] $(date --iso-8601=seconds) starting ${label}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_TIMEOUT=15m \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        PROFILE_STEP_START="${PROFILE_STEP_START}" \
        PROFILE_STEP_END="${PROFILE_STEP_END}" \
        PROFILE_RANKS="${PROFILE_RANKS}" \
        bash "${wrapper}"
    echo "[a3b-postwarm-ab] $(date --iso-8601=seconds) completed ${label}"
}

run_case healthy "${HEALTHY_WRAPPER}"
run_case nep "${NEP_WRAPPER}"

HEALTHY_NAME="a3b_30b_original_128e_ep16_ep16_flex_hybridep_noopt_i${TRAIN_ITERS}"
NEP_NAME="a3b_30b_original_128e_ep16_ep12_flex_hybridep_noopt_i${TRAIN_ITERS}"
HEALTHY_RUN="${ROOT_DIR}/healthy/${HEALTHY_NAME}/${SLURM_JOB_ID}"
NEP_RUN="${ROOT_DIR}/nep/${NEP_NAME}/${SLURM_JOB_ID}"
COMPARISON_JSON="${ROOT_DIR}/timing_comparison_${SLURM_JOB_ID}.json"
TRACE_ANALYSIS_JSON="${ROOT_DIR}/trace_analysis_${SLURM_JOB_ID}.json"

python3 - "${HEALTHY_RUN}/driver_${SLURM_JOB_ID}.log" \
    "${NEP_RUN}/driver_${SLURM_JOB_ID}.log" "${COMPARISON_JSON}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

pattern = re.compile(
    r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)"
)
result = {}
for label, path in (("healthy", Path(sys.argv[1])), ("nep", Path(sys.argv[2]))):
    rows = [
        {"iteration": int(i), "elapsed_ms": float(ms), "tflops_per_gpu": float(tf)}
        for i, ms, tf in pattern.findall(path.read_text())
        if int(i) >= 8
    ]
    if [row["iteration"] for row in rows] != [8, 9, 10]:
        raise RuntimeError(f"{label}: expected clean iterations 8-10, got {rows}")
    result[label] = {
        "rows": rows,
        "mean_elapsed_ms": statistics.mean(row["elapsed_ms"] for row in rows),
        "std_elapsed_ms": statistics.stdev(row["elapsed_ms"] for row in rows),
        "mean_tflops_per_gpu": statistics.mean(row["tflops_per_gpu"] for row in rows),
    }
healthy = result["healthy"]["mean_elapsed_ms"]
nep = result["nep"]["mean_elapsed_ms"]
result["comparison"] = {
    "latency_gap_ms": nep - healthy,
    "nep_slowdown_percent": 100.0 * (nep / healthy - 1.0),
    "owner_time_parity_percent": 100.0 * healthy / nep,
    "tflops_parity_percent": 100.0
    * result["nep"]["mean_tflops_per_gpu"]
    / result["healthy"]["mean_tflops_per_gpu"],
}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result["comparison"], sort_keys=True))
PY

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

timeout --foreground --signal=TERM --kill-after=30s 10m \
    srun --overlap --nodes=1 --nodelist="${FIRST_NODE}" --ntasks=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc "python '${ANALYZER}' \\
        --healthy-trace-dir '${HEALTHY_RUN}/torch_profile' \\
        --nep-trace-dir '${NEP_RUN}/torch_profile' \\
        --ranks 0 16 --steps '${PROFILE_STEPS}' --output '${TRACE_ANALYSIS_JSON}'"

echo "[a3b-postwarm-ab] complete timing=${COMPARISON_JSON} traces=${TRACE_ANALYSIS_JSON}"
