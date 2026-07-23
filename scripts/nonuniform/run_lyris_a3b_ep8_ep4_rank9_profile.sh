#!/bin/bash

# Collect reduced-replica CPU stacks, shapes, and execution traces around the
# rank-9 GroupedLinear backward delay in a placement-controlled A/B/A run.

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
#SBATCH --job-name=coreai_comparch_sysarch-nep.rank9-profile

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_rank9_profile}"
WRAPPER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_post_graph_host_aba.sh"

env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${ROOT_DIR}" \
    IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}" \
    CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}" \
    CASE_INCLUDE_HEALTHY=0 \
    CASE_TRAIN_ITERS=6 \
    CASE_LR_WSD_DECAY_ITERS=3 \
    CASE_TIMEOUT=8m \
    CASE_PROFILE_STEP_START=3 \
    CASE_PROFILE_STEP_END=5 \
    CASE_PROFILE_RANKS="8 9 10 11" \
    CASE_EXTRA_MEGATRON_ARGS="--pytorch-profiler-collect-shapes --pytorch-profiler-collect-callstack --pytorch-profiler-collect-chakra" \
    bash "${WRAPPER}"
