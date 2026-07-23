#!/bin/bash

# Clean top-k sweep with adjacent healthy/NEP pairs and all-rank traces.

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
#SBATCH --job-name=coreai_comparch_sysarch-nep.mbs1-topk-clean

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_mbs1_topk_clean_sweep}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-10m}"

run_pair() {
    local label="$1"
    local topk="$2"
    local initial_skip_preflight="$3"

    echo "[ep8-ep4-mbs1-topk-clean] $(date --iso-8601=seconds) starting ${label} top-k ${topk}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=pair \
        CASE_TIMEOUT="${CASE_TIMEOUT}" \
        CASE_PROFILE=1 \
        CASE_TRAIN_ITERS=16 \
        CASE_LR_WSD_DECAY_ITERS=4 \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        INITIAL_SKIP_PREFLIGHT="${initial_skip_preflight}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        MOE_ROUTER_TOPK="${topk}" \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_BIASED= \
        bash "${SWEEP}"
    echo "[ep8-ep4-mbs1-topk-clean] $(date --iso-8601=seconds) completed ${label} top-k ${topk}"
}

run_pair topk2_first 2 0
run_pair topk4 4 1
run_pair topk6 6 1
run_pair topk8 8 1
run_pair topk12 12 1
run_pair topk2_repeat 2 1
