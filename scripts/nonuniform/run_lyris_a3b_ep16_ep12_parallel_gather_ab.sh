#!/bin/bash

# Multi-owner EP16/EP12 serial/parallel Gather submission comparison.

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
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep16-12-par-gather

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep16_ep12_parallel_gather_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep16_ep12_scatter_progress_ab.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

run_stage() {
    local stage="$1"
    local window="$2"
    local timing="$3"
    local profile="$4"

    echo "[ep16-parallel-gather] $(date --iso-8601=seconds) starting ${stage} window=${window}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${stage}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        RUN_CORRECTNESS=0 \
        RUN_HEALTHY_TIMING=0 \
        RUN_NEP_TIMING="${timing}" \
        RUN_HEALTHY_PROFILE=0 \
        RUN_NEP_PROFILE="${profile}" \
        PARALLEL_GATHER_WINDOW="${window}" \
        TIMING_ITERS=10 \
        PROFILE_ITERS=8 \
        CASE_TIMEOUT=10m \
        bash "${RUNNER}"
    echo "[ep16-parallel-gather] $(date --iso-8601=seconds) completed ${stage}"
}

echo "[ep16-parallel-gather] $(date --iso-8601=seconds) starting correctness"
env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${ROOT_DIR}/correctness" \
    IMAGE="${IMAGE}" \
    CONTAINER_NAME="${CONTAINER_NAME}" \
    RUN_CORRECTNESS=1 \
    RUN_HEALTHY_TIMING=0 \
    RUN_NEP_TIMING=0 \
    RUN_HEALTHY_PROFILE=0 \
    RUN_NEP_PROFILE=0 \
    PARALLEL_GATHER_WINDOW=2 \
    bash "${RUNNER}"

run_stage timing_serial 1 1 0
run_stage timing_parallel2 2 1 0
run_stage timing_serial_repeat 1 1 0
run_stage profile_serial 1 0 1
run_stage profile_parallel2 2 0 1

echo "[ep16-parallel-gather] $(date --iso-8601=seconds) complete"
