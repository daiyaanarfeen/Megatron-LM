#!/bin/bash

# Controlled low-compute-window top-k A-B-A test in one allocation.

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
#SBATCH --job-name=coreai_comparch_sysarch-nep.mbs1-topk-aba

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_mbs1_topk_aba}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-10m}"
TOPK_A="${TOPK_A:-6}"
TOPK_B="${TOPK_B:-8}"
PAIR_PROFILE="${PAIR_PROFILE:-1}"
PAIR_TRAIN_ITERS="${PAIR_TRAIN_ITERS:-8}"
PAIR_LR_WSD_DECAY_ITERS="${PAIR_LR_WSD_DECAY_ITERS:-4}"

for topk in "${TOPK_A}" "${TOPK_B}"; do
    if ! [[ "${topk}" =~ ^[1-9][0-9]*$ ]] || ((topk > 128)); then
        echo "TOPK_A and TOPK_B must be integers in [1, 128]" >&2
        exit 2
    fi
done

run_pair() {
    local label="$1"
    local topk="$2"
    local initial_skip_preflight="$3"

    echo "[ep8-ep4-mbs1-topk-aba] $(date --iso-8601=seconds) starting ${label} top-k ${topk}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=pair \
        CASE_TIMEOUT="${CASE_TIMEOUT}" \
        CASE_PROFILE="${PAIR_PROFILE}" \
        CASE_TRAIN_ITERS="${PAIR_TRAIN_ITERS}" \
        CASE_LR_WSD_DECAY_ITERS="${PAIR_LR_WSD_DECAY_ITERS}" \
        INITIAL_SKIP_PREFLIGHT="${initial_skip_preflight}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        MOE_ROUTER_TOPK="${topk}" \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_BIASED= \
        bash "${SWEEP}"
    echo "[ep8-ep4-mbs1-topk-aba] $(date --iso-8601=seconds) completed ${label} top-k ${topk}"
}

run_pair "topk${TOPK_A}_first" "${TOPK_A}" 0
run_pair "topk${TOPK_B}" "${TOPK_B}" 1
run_pair "topk${TOPK_A}_repeat" "${TOPK_A}" 1
