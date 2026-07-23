#!/bin/bash

# Correctness gate plus compute-heavy healthy/NEP comparison for prompt Scatter progress.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.scatter-progress

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_scatter_progress_ab}"
CORRECTNESS="${REPO_DIR}/scripts/nonuniform/run_lyris_ep8_ep4_split_correctness.sh"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
FULL_MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE:-2}"
REDUCED_MICRO_BATCH_SIZE="${REDUCED_MICRO_BATCH_SIZE:-1}"
SCATTER_CHUNKS="${SCATTER_CHUNKS:-2}"
TIMING_ITERS="${TIMING_ITERS:-18}"
PROFILE_ITERS="${PROFILE_ITERS:-8}"

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
    python -m isort --check-only \
        megatron/core/distributed/nonuniform_ep.py \
        tests/unit_tests/distributed/test_nonuniform_ep.py &&
    python -m py_compile \
        megatron/core/distributed/nonuniform_ep.py \
        tests/unit_tests/distributed/test_nonuniform_ep.py &&
    python -m pytest -q \
        tests/unit_tests/distributed/test_nonuniform_ep.py \
        tests/unit_tests/tensor_parallel/test_mappings.py \
        -k 'scatter_chunk or scatter_work_defers or split_host_phases_defer_edp_and_scatter or model_ep_a2a_burst_end or scatter_progress or all_to_all_burst_callbacks'
"

echo "[scatter-progress] $(date --iso-8601=seconds) starting gradient correctness gate"
env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${ROOT_DIR}/correctness" \
    IMAGE="${IMAGE}" \
    CONTAINER_NAME="${CONTAINER_NAME}" \
    CASE_NUM_LAYERS=2 \
    REFERENCE_SCATTER_CHUNKS=1 \
    SPLIT_SCATTER_CHUNKS=4 \
    SPLIT_A2A_SCATTER_SCHEDULER=1 \
    CASE_TIMEOUT=6m \
    bash "${CORRECTNESS}"

run_pair() {
    local stage="$1"
    local profile="$2"
    local train_iters="$3"
    local profile_start="$4"
    local profile_end="$5"

    echo "[scatter-progress] $(date --iso-8601=seconds) starting ${stage}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${stage}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=pair \
        CASE_PROFILE="${profile}" \
        CASE_PROFILE_STEP_START="${profile_start}" \
        CASE_PROFILE_STEP_END="${profile_end}" \
        CASE_PROFILE_RANKS=all \
        CASE_TRAIN_ITERS="${train_iters}" \
        CASE_LR_WSD_DECAY_ITERS="$((train_iters / 2))" \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_LOG_ENERGY=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_DEFER_MODEL_EP_FENCE=1 \
        CASE_A2A_SCATTER_SCHEDULER=1 \
        CASE_SKIP_OWNER_GRAD_CHECK=0 \
        CASE_SEQ_LENGTH="${SEQ_LENGTH}" \
        CASE_EXTRA_MEGATRON_ARGS="--moe-router-bias-update-rate 0.0" \
        CASE_TIMEOUT=10m \
        INITIAL_SKIP_PREFLIGHT=1 \
        PAIR_MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE}" \
        PAIR_NUM_MICROBATCHES=1 \
        PAIR_REDUCED_MICRO_BATCH_SIZE="${REDUCED_MICRO_BATCH_SIZE}" \
        PAIR_REDUCED_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING=0 \
        MOE_ROUTER_FORCE_UNIFORM_ROUTING=1 \
        MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
        MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS="${SCATTER_CHUNKS}" \
        bash "${SWEEP}"
    echo "[scatter-progress] $(date --iso-8601=seconds) completed ${stage}"
}

run_pair timing 0 "${TIMING_ITERS}" 6 7
run_pair profile 1 "${PROFILE_ITERS}" 5 7

echo "[scatter-progress] $(date --iso-8601=seconds) complete"
