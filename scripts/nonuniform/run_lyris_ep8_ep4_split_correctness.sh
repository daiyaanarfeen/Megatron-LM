#!/bin/bash

# Two-layer numerical gate for stable versus split-phase EP8/EP4.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=3
#SBATCH --segment=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-split-ab-correct

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_ep8_ep4_split_correctness}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_nonuniform_ep_overlap_smoke.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-6m}"
SPLIT_TARGET_CHUNKS="${SPLIT_TARGET_CHUNKS:-2}"
SPLIT_ASYNC_CHUNK_WINDOW="${SPLIT_ASYNC_CHUNK_WINDOW:-16}"
REFERENCE_SCATTER_CHUNKS="${REFERENCE_SCATTER_CHUNKS:-1}"
SPLIT_SCATTER_CHUNKS="${SPLIT_SCATTER_CHUNKS:-2}"
REFERENCE_APPROACH="${REFERENCE_APPROACH:-nccl}"
SPLIT_APPROACH="${SPLIT_APPROACH:-nccl}"
REFERENCE_A2A_SCATTER_SCHEDULER="${REFERENCE_A2A_SCATTER_SCHEDULER:-0}"
SPLIT_A2A_SCATTER_SCHEDULER="${SPLIT_A2A_SCATTER_SCHEDULER:-1}"
CASE_NUM_LAYERS="${CASE_NUM_LAYERS:-2}"
CASE_LABEL="${CASE_LABEL:-ep8_ep4}"
CASE_DISPLAY="${CASE_DISPLAY:-ep8-4}"
CASE_RUN_NNODES="${CASE_RUN_NNODES:-3}"
CASE_RUN_NPROC_PER_NODE="${CASE_RUN_NPROC_PER_NODE:-4}"
CASE_RUN_WORLD_SIZE="${CASE_RUN_WORLD_SIZE:-$((CASE_RUN_NNODES * CASE_RUN_NPROC_PER_NODE))}"
CASE_USE_DIRECT_SRUN_RANKS="${CASE_USE_DIRECT_SRUN_RANKS:-0}"
CASE_GLOBAL_BATCH_SIZE="${CASE_GLOBAL_BATCH_SIZE:-24}"
CASE_SEQ_LENGTH="${CASE_SEQ_LENGTH:-128}"
CASE_NUM_EXPERTS="${CASE_NUM_EXPERTS:-8}"
CASE_TENSOR_MODEL_PARALLEL_SIZE="${CASE_TENSOR_MODEL_PARALLEL_SIZE:-2}"
CASE_EXPERT_MODEL_PARALLEL_SIZE="${CASE_EXPERT_MODEL_PARALLEL_SIZE:-8}"
CASE_NONUNIFORM_EP_TOPOLOGY="${CASE_NONUNIFORM_EP_TOPOLOGY:-4 2}"
CASE_DEBUG_RANKS="${CASE_DEBUG_RANKS:-0 4 8}"
CASE_EXPECTED_RANKS="${CASE_EXPECTED_RANKS:-12}"
REFERENCE_SPLIT_HOST_PHASES="${REFERENCE_SPLIT_HOST_PHASES:-0}"
REFERENCE_POST_GRAPH_HOST_PHASES="${REFERENCE_POST_GRAPH_HOST_PHASES:-0}"
SPLIT_POST_GRAPH_HOST_PHASES="${SPLIT_POST_GRAPH_HOST_PHASES:-0}"
REFERENCE_EXPERT_BUCKET_GROUPS="${REFERENCE_EXPERT_BUCKET_GROUPS:-12}"
SPLIT_EXPERT_BUCKET_GROUPS="${SPLIT_EXPERT_BUCKET_GROUPS:-3}"
STABLE_NAME="${CASE_LABEL}_${REFERENCE_APPROACH}_l${CASE_NUM_LAYERS}_groups${REFERENCE_EXPERT_BUCKET_GROUPS}_schunks${REFERENCE_SCATTER_CHUNKS}_checksum_${SLURM_JOB_ID}"
SPLIT_NAME="${CASE_LABEL}_${SPLIT_APPROACH}_split_l${CASE_NUM_LAYERS}_groups${SPLIT_EXPERT_BUCKET_GROUPS}_chunks${SPLIT_TARGET_CHUNKS}_schunks${SPLIT_SCATTER_CHUNKS}_checksum_${SLURM_JOB_ID}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" bash -lc "
    cd '${REPO_DIR}' &&
    python -m black --required-version 26 --check megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py &&
    python -m isort --check-only megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py &&
    python -m py_compile megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py &&
    python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py tests/unit_tests/tensor_parallel/test_mappings.py -k 'scatter_chunk or scatter_queue or scatter_work_defers or ready_gate or split_host_phases_defer_edp_and_scatter or pipelined_host_phases or a2a_scatter_scheduler_preserves or model_ep_a2a_burst or scatter_progress or scatter_submission or all_to_all_burst_callbacks'
"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

run_case() {
    local name="$1"
    local split_host_phases="$2"
    local debug="$3"
    local master_port="$4"
    local target_chunks="$5"
    local async_chunk_window="$6"
    local ddp_approach="$7"
    local expert_bucket_groups="$8"
    local post_graph_host_phases="$9"
    local scatter_chunks="${10}"
    local a2a_scatter_scheduler="${11}"
    local checksum_dir="${ROOT_DIR}/${name}/checksums"

    echo "[${CASE_DISPLAY}-split-correctness] $(date --iso-8601=seconds) starting ${name}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${name}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${CASE_RUN_NNODES}" \
            RUN_NPROC_PER_NODE="${CASE_RUN_NPROC_PER_NODE}" \
            RUN_WORLD_SIZE="${CASE_RUN_WORLD_SIZE}" \
            USE_DIRECT_SRUN_RANKS="${CASE_USE_DIRECT_SRUN_RANKS}" \
            RUN_PREFLIGHT_TESTS=0 \
            ENABLE_PYTORCH_PROFILER=0 \
            TRAIN_ITERS=2 \
            GLOBAL_BATCH_SIZE="${CASE_GLOBAL_BATCH_SIZE}" \
            MICRO_BATCH_SIZE=1 \
            NUM_LAYERS="${CASE_NUM_LAYERS}" \
            HIDDEN_SIZE=256 \
            FFN_HIDDEN_SIZE=1024 \
            NUM_ATTENTION_HEADS=4 \
            SEQ_LENGTH="${CASE_SEQ_LENGTH}" \
            NUM_EXPERTS="${CASE_NUM_EXPERTS}" \
            TENSOR_MODEL_PARALLEL_SIZE="${CASE_TENSOR_MODEL_PARALLEL_SIZE}" \
            EXPERT_MODEL_PARALLEL_SIZE="${CASE_EXPERT_MODEL_PARALLEL_SIZE}" \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            NONUNIFORM_MODE=ep \
            NONUNIFORM_EP_DDP_APPROACH="${ddp_approach}" \
            NONUNIFORM_EP_TOPOLOGY="${CASE_NONUNIFORM_EP_TOPOLOGY}" \
            USE_GLOO_PROCESS_GROUPS=1 \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=1 \
            TORCH_NCCL_BLOCKING_WAIT=0 \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES="${post_graph_host_phases}" \
            MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE="${a2a_scatter_scheduler}" \
            MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER="${a2a_scatter_scheduler}" \
            MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS="${target_chunks}" \
            MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS="${scatter_chunks}" \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW="${async_chunk_window}" \
            MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${expert_bucket_groups}" \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            MEGATRON_NONUNIFORM_EP_DEBUG="${debug}" \
            MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="${CASE_DEBUG_RANKS}" \
            EXTRA_MEGATRON_ARGS="--nonuniform-log-grad-checksum --nonuniform-grad-checksum-dir ${checksum_dir} --attention-dropout 0.0 --hidden-dropout 0.0 --cuda-graph-impl local --cuda-graph-modules moe_router --te-rng-tracker --no-load-rng --distributed-timeout-minutes 3" \
            bash "${RUNNER}"
    echo "[${CASE_DISPLAY}-split-correctness] $(date --iso-8601=seconds) completed ${name}"
}

run_case "${STABLE_NAME}" "${REFERENCE_SPLIT_HOST_PHASES}" 0 29931 "${SPLIT_TARGET_CHUNKS}" "${SPLIT_ASYNC_CHUNK_WINDOW}" "${REFERENCE_APPROACH}" "${REFERENCE_EXPERT_BUCKET_GROUPS}" "${REFERENCE_POST_GRAPH_HOST_PHASES}" "${REFERENCE_SCATTER_CHUNKS}" "${REFERENCE_A2A_SCATTER_SCHEDULER}"
run_case "${SPLIT_NAME}" 1 1 29932 "${SPLIT_TARGET_CHUNKS}" "${SPLIT_ASYNC_CHUNK_WINDOW}" "${SPLIT_APPROACH}" "${SPLIT_EXPERT_BUCKET_GROUPS}" "${SPLIT_POST_GRAPH_HOST_PHASES}" "${SPLIT_SCATTER_CHUNKS}" "${SPLIT_A2A_SCATTER_SCHEDULER}"

python3 - \
    "${ROOT_DIR}/${STABLE_NAME}/checksums" \
    "${ROOT_DIR}/${SPLIT_NAME}/checksums" \
    "${ROOT_DIR}/${STABLE_NAME}/driver_${SLURM_JOB_ID}.log" \
    "${ROOT_DIR}/${SPLIT_NAME}/driver_${SLURM_JOB_ID}.log" \
    "${CASE_EXPECTED_RANKS}" \
    "${CASE_DISPLAY}" <<'PY'
import math
import re
import statistics
import sys
from pathlib import Path

NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FIELDS = (
    "weighted_sum",
    "weighted_abs",
    "weighted_sq",
    "weighted_numel",
    "dense_weighted_sum",
    "dense_weighted_abs",
    "dense_weighted_sq",
    "dense_weighted_numel",
    "dense_numel",
    "expert_weighted_sum",
    "expert_weighted_abs",
    "expert_weighted_sq",
    "expert_weighted_numel",
    "expert_numel",
)
PATTERN = re.compile(
    r"\[nonuniform-grad-checksum\] iteration=(\S+) rank=(\d+) "
    + " ".join(rf"{field}=({NUMBER})" for field in FIELDS)
)


def load(checksum_dir, driver_path):
    records = {}
    checksum_paths = sorted(Path(checksum_dir).glob("rank_*.log"))
    for path in checksum_paths:
        for match in PATTERN.finditer(path.read_text()):
            iteration = match.group(1)
            rank = int(match.group(2))
            records.setdefault(iteration, {})[rank] = tuple(
                float(value) for value in match.groups()[2:]
            )
    if not records:
        raise RuntimeError(f"No gradient checksums found in {checksum_dir}")
    driver_text = Path(driver_path).read_text()
    if len(re.findall(r"number of nan iterations:\s+0", driver_text)) < 2:
        raise RuntimeError(f"Missing two finite iterations in {driver_path}")
    return driver_text, records


stable_text, stable = load(sys.argv[1], sys.argv[3])
split_text, split = load(sys.argv[2], sys.argv[4])
expected_ranks = int(sys.argv[5])
display = sys.argv[6]
if "submit split_dispatch_gather_owner_barrier" not in split_text:
    raise RuntimeError("Split-host scheduler did not emit its Gather-phase launch marker")
if "split_host_phases_fallback" in split_text:
    raise RuntimeError("Split-host scheduler fell back to the stable inline path")

common_iterations = set(stable).intersection(split)
if not common_iterations:
    raise RuntimeError("Stable and split runs have no common checksum iteration")
iteration = max(common_iterations, key=lambda value: int(value))
stable_rows = list(stable[iteration].values())
split_rows = list(split[iteration].values())
if len(stable_rows) != expected_ranks or len(split_rows) != expected_ranks:
    raise RuntimeError(
        f"Expected {expected_ranks} stable and split ranks, got "
        f"{len(stable_rows)} and {len(split_rows)}"
    )

rtol = 1.0e-6
atol = 1.0e-8
exact_fields = {
    "weighted_numel",
    "dense_weighted_numel",
    "dense_numel",
    "expert_weighted_numel",
    "expert_numel",
}
for index, label in enumerate(FIELDS):
    stable_median = statistics.median(row[index] for row in stable_rows)
    split_median = statistics.median(row[index] for row in split_rows)
    scale = max(abs(stable_median), abs(split_median), 1.0)
    stable_spread = max(abs(row[index] - stable_median) for row in stable_rows) / scale
    split_spread = max(abs(row[index] - split_median) for row in split_rows) / scale
    relative_delta = abs(stable_median - split_median) / scale
    print(
        f"[{display}-split-correctness] iteration={iteration} {label} "
        f"stable={stable_median:.17e} split={split_median:.17e} "
        f"relative_delta={relative_delta:.3e} "
        f"stable_spread={stable_spread:.3e} split_spread={split_spread:.3e}"
    )
    if label in exact_fields:
        if stable_spread != 0.0 or split_spread != 0.0:
            raise RuntimeError(f"Per-rank coverage disagreement for {label}")
        if stable_median != split_median:
            raise RuntimeError(f"Coverage mismatch for {label}")
    else:
        if stable_spread > rtol or split_spread > rtol:
            raise RuntimeError(f"Per-rank checksum disagreement for {label}")
        if not math.isclose(stable_median, split_median, rel_tol=rtol, abs_tol=atol):
            raise RuntimeError(f"Gradient checksum mismatch for {label}")

print(f"[{display}-split-correctness] PASS")
PY
