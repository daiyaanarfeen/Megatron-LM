#!/bin/bash

# Run the a500 healthy/NEP readiness comparison twice on one allocation so
# changing only the reduced replica's accumulation count is causal.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:20:00
#SBATCH --segment=4
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail

BENCH_REPO="${BENCH_REPO:-/home/darfeen/Megatron-LM}"
RUNNER="${BENCH_REPO}/scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh"
ROOT="${ROOT:-/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_readiness_controlled}"

common_env=(
    GROUP=ep8
    WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
    TRAIN_ITERS=10
    PROFILE_STEP_START=5
    PROFILE_STEP_END=7
    TIMING_START=8
    HEALTHY_NUM_MICROBATCHES=12
    NEP_FULL_NUM_MICROBATCHES=12
)

env "${common_env[@]}" \
    ROOT_DIR="${ROOT}/ga12_12" \
    NEP_REDUCED_NUM_MICROBATCHES=12 \
    bash "${RUNNER}"

env "${common_env[@]}" \
    ROOT_DIR="${ROOT}/ga12_9" \
    NEP_REDUCED_NUM_MICROBATCHES=9 \
    bash "${RUNNER}"
