#!/bin/bash

# Correctness plus same-allocation serial/parallel Gather submission comparison.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-par-gather

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_parallel_gather_ab}"
CORRECTNESS="${REPO_DIR}/scripts/nonuniform/run_lyris_ep8_ep4_split_correctness.sh"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
TIMING_ITERS="${TIMING_ITERS:-12}"
PROFILE_ITERS="${PROFILE_ITERS:-8}"

run_nep() {
    local stage="$1"
    local gather_window="$2"
    local profile="$3"
    local train_iters="$4"

    echo "[parallel-gather] $(date --iso-8601=seconds) starting ${stage} window=${gather_window}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${stage}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=proportional \
        CASE_PROFILE="${profile}" \
        CASE_PROFILE_STEP_START=5 \
        CASE_PROFILE_STEP_END=7 \
        CASE_PROFILE_RANKS=all \
        CASE_TRAIN_ITERS="${train_iters}" \
        CASE_LR_WSD_DECAY_ITERS="$((train_iters / 2))" \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_LOG_ENERGY=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_DEFER_MODEL_EP_FENCE=1 \
        CASE_A2A_SCATTER_SCHEDULER=1 \
        CASE_PARALLEL_GATHER_WINDOW="${gather_window}" \
        CASE_SKIP_OWNER_GRAD_CHECK=0 \
        CASE_SEQ_LENGTH="${SEQ_LENGTH}" \
        CASE_EXTRA_MEGATRON_ARGS="--moe-router-bias-update-rate 0.0" \
        CASE_TIMEOUT=8m \
        INITIAL_SKIP_PREFLIGHT=1 \
        PAIR_MICRO_BATCH_SIZE=2 \
        PAIR_NUM_MICROBATCHES=1 \
        PAIR_REDUCED_MICRO_BATCH_SIZE=1 \
        PAIR_REDUCED_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING=0 \
        MOE_ROUTER_FORCE_UNIFORM_ROUTING=1 \
        MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
        MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=2 \
        bash "${SWEEP}"
    echo "[parallel-gather] $(date --iso-8601=seconds) completed ${stage}"
}

echo "[parallel-gather] $(date --iso-8601=seconds) starting correctness"
env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${ROOT_DIR}/correctness" \
    IMAGE="${IMAGE}" \
    CONTAINER_NAME="${CONTAINER_NAME}" \
    CASE_NUM_LAYERS=2 \
    REFERENCE_SCATTER_CHUNKS=1 \
    SPLIT_SCATTER_CHUNKS=2 \
    SPLIT_A2A_SCATTER_SCHEDULER=1 \
    SPLIT_PARALLEL_GATHER_WINDOW=2 \
    CASE_TIMEOUT=6m \
    bash "${CORRECTNESS}"

run_nep timing_serial 1 0 "${TIMING_ITERS}"
run_nep timing_parallel2 2 0 "${TIMING_ITERS}"
run_nep timing_serial_repeat 1 0 "${TIMING_ITERS}"
run_nep profile_serial 1 1 "${PROFILE_ITERS}"
run_nep profile_parallel2 2 1 "${PROFILE_ITERS}"

echo "[parallel-gather] $(date --iso-8601=seconds) complete"
