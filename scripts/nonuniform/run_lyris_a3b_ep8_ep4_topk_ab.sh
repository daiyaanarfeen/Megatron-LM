#!/bin/bash

# Same-allocation expert-GEMM shape test: top-k 6 versus top-k 8.

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
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-topk-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_topk_ab}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-10m}"

run_pair() {
    local topk="$1"
    local initial_skip_preflight="$2"

    echo "[ep8-ep4-topk-ab] $(date --iso-8601=seconds) starting top-k ${topk}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/topk${topk}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=pair \
        CASE_TIMEOUT="${CASE_TIMEOUT}" \
        CASE_PROFILE=1 \
        INITIAL_SKIP_PREFLIGHT="${initial_skip_preflight}" \
        MOE_ROUTER_TOPK="${topk}" \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_BIASED= \
        bash "${SWEEP}"
    echo "[ep8-ep4-topk-ab] $(date --iso-8601=seconds) completed top-k ${topk}"
}

run_pair 6 0
run_pair 8 1
