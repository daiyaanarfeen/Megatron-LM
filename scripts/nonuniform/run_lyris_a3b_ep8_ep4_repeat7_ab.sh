#!/bin/bash

# Seven-layer original-repeat healthy versus ordered NEP profiler comparison.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b-ep8-4-repeat7-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
export REPO_DIR
export ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_repeat7_ab}"
export CASE_LABEL="repeat7"
export CASE_HYBRID_LAYER_PATTERN="MEMEM*E"
export CASE_SELECTION="healthy_split"
export CASE_TRAIN_ITERS="10"
export CASE_TIMEOUT="8m"
export CASE_EXIT_DURATION_IN_MINS="7"

exec bash "${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_split_l4_ab.sh"
