#!/bin/bash

# Matched EP8/EP4 routing-balance sensitivity test.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-router-balance

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_router_balance}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
FORCED_BIAS_STD="${FORCED_BIAS_STD:--1.0}"

run_case() {
    local routing="$1"
    local selection="$2"
    local force_balance="$3"
    local force_bias="$4"

    echo "[ep8-ep4-router-balance] $(date --iso-8601=seconds) routing=${routing} case=${selection}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${routing}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION="${selection}" \
        CASE_PROFILE=1 \
        MOE_ROUTER_FORCE_LOAD_BALANCING="${force_balance}" \
        MOE_ROUTER_FORCE_BIASED="${force_bias}" \
        bash "${SWEEP}"
}

run_case balanced healthy 1 ""
run_case balanced proportional 1 ""
run_case biased healthy 0 "${FORCED_BIAS_STD}"
run_case biased proportional 0 "${FORCED_BIAS_STD}"
