#!/bin/bash

# Two-layer numerical gate for stable versus split-phase EP16/EP12.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=7
#SBATCH --segment=7
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep16-12-split-correct

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"

env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_ep16_ep12_split_correctness}" \
    CASE_LABEL=ep16_ep12 \
    CASE_DISPLAY=ep16-12 \
    CASE_RUN_NNODES=7 \
    CASE_RUN_NPROC_PER_NODE=4 \
    CASE_GLOBAL_BATCH_SIZE=28 \
    CASE_NUM_LAYERS="${CASE_NUM_LAYERS:-2}" \
    CASE_NUM_EXPERTS=48 \
    CASE_TENSOR_MODEL_PARALLEL_SIZE=2 \
    CASE_EXPERT_MODEL_PARALLEL_SIZE=16 \
    CASE_NONUNIFORM_EP_TOPOLOGY="8 6" \
    CASE_DEBUG_RANKS="0 12 16 24" \
    CASE_EXPECTED_RANKS=28 \
    REFERENCE_SCATTER_CHUNKS=1 \
    SPLIT_SCATTER_CHUNKS="${SPLIT_SCATTER_CHUNKS:-4}" \
    SPLIT_A2A_SCATTER_SCHEDULER=1 \
    SPLIT_ASYNC_CHUNK_WINDOW=64 \
    CASE_TIMEOUT="${CASE_TIMEOUT:-7m}" \
    bash "${REPO_DIR}/scripts/nonuniform/run_lyris_ep8_ep4_split_correctness.sh"
