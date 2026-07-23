#!/bin/bash

# Same-allocation healthy and ordered-split owner chunk-count comparison.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ordered-r14-chunks

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_repeat14_chunk_sweep}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_split_l4_ab.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TARGET_CHUNKS_LIST="${TARGET_CHUNKS_LIST:-1 2 4}"

common_env=(
    REPO_DIR="${REPO_DIR}"
    ROOT_DIR="${ROOT_DIR}"
    IMAGE="${IMAGE}"
    CONTAINER_NAME="${CONTAINER_NAME}"
    CASE_HYBRID_LAYER_PATTERN="MEMEM*EMEMEM*E"
    CASE_TRAIN_ITERS=7
    CASE_TIMEOUT=7m
    CASE_EXIT_DURATION_IN_MINS=6
    CASE_MICRO_BATCH_SIZE=1
    CASE_HEALTHY_GLOBAL_BATCH_SIZE=8
    CASE_NEP_GLOBAL_BATCH_SIZE=6
    CASE_PROFILE=1
    CASE_PROFILE_STEP_START=3
    CASE_PROFILE_STEP_END=5
    CASE_PROFILE_RANKS=0
)

env "${common_env[@]}" \
    CASE_LABEL=ordered_repeat14_chunk_sweep_healthy_gb300 \
    CASE_SELECTION=healthy \
    bash "${RUNNER}"

env "${common_env[@]}" \
    CASE_LABEL=ordered_repeat14_original_gb300 \
    CASE_SELECTION=split \
    CASE_SPLIT_TARGET_CHUNKS= \
    CASE_SPLIT_ASYNC_CHUNK_WINDOW=16 \
    bash "${RUNNER}"

for target_chunks in ${TARGET_CHUNKS_LIST}; do
    async_chunk_window=$((8 * target_chunks))
    if ((async_chunk_window < 16)); then
        async_chunk_window=16
    fi
    env "${common_env[@]}" \
        CASE_LABEL="ordered_repeat14_chunks${target_chunks}_gb300" \
        CASE_SELECTION=split \
        CASE_SPLIT_TARGET_CHUNKS="${target_chunks}" \
        CASE_SPLIT_ASYNC_CHUNK_WINDOW="${async_chunk_window}" \
        bash "${RUNNER}"
done
