#!/bin/bash

# Graph-enabled counterpart to the short eager EP64/64 versus EP64/48 A/B.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=32
#SBATCH --segment=16
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b192e-graph-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_192e_ep64_ep48_ab.sh"

env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${REPO_DIR}/slurm_runs/lyris_a3b_192e_ep64_ep48_graph_ab" \
    CASE_TIMEOUT=14m \
    CASE_TRAIN_ITERS=8 \
    CASE_LR_WSD_DECAY_ITERS=4 \
    CASE_CUDA_GRAPH_IMPL=local \
    CASE_PROFILE_STEP_START=4 \
    CASE_PROFILE_STEP_END=5 \
    CASE_EXIT_DURATION_IN_MINS=13 \
    CASE_NAME_SUFFIX=_graph \
    bash "${RUNNER}"
