#!/bin/bash

# Same-allocation ordered-split comparison at MBS1 and MBS2.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ordered-r14-mbs-overlap

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_repeat14_mbs_overlap_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_split_l4_ab.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

run_pair() {
    local label="$1"
    local micro_batch_size="$2"
    local healthy_global_batch_size="$3"
    local nep_global_batch_size="$4"

    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_LABEL="${label}" \
        CASE_HYBRID_LAYER_PATTERN="MEMEM*EMEMEM*E" \
        CASE_SELECTION=healthy_split \
        CASE_TRAIN_ITERS=7 \
        CASE_TIMEOUT=7m \
        CASE_EXIT_DURATION_IN_MINS=6 \
        CASE_MICRO_BATCH_SIZE="${micro_batch_size}" \
        CASE_HEALTHY_GLOBAL_BATCH_SIZE="${healthy_global_batch_size}" \
        CASE_NEP_GLOBAL_BATCH_SIZE="${nep_global_batch_size}" \
        CASE_PROFILE=1 \
        CASE_PROFILE_STEP_START=3 \
        CASE_PROFILE_STEP_END=5 \
        CASE_PROFILE_RANKS=0 \
        bash "${RUNNER}"
}

run_pair ordered_repeat14_mbs1_gb300 1 8 6
run_pair ordered_repeat14_mbs2_gb300 2 16 12
