#!/bin/bash
# Two rounds on one allocation, reversing both workload and healthy/NEP order.
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
export GROUP=ep8
export WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
export TRAIN_ITERS=35 PROFILE_STEP_START=32 PROFILE_STEP_END=34
export TIMING_START=11 TIMING_END=30
export HEALTHY_NUM_MICROBATCHES=12 NEP_FULL_NUM_MICROBATCHES=12
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
RUNNER=/home/darfeen/Megatron-LM/scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_readiness_confirm
git -C "${CODE_REPO}" rev-parse HEAD
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
read -r -a rounds <<< "${CONFIRM_ROUNDS:-1 2}"
for round in "${rounds[@]}"; do
    [[ "${round}" == 1 || "${round}" == 2 ]] || exit 2
    if [[ "${round}" == 1 ]]; then
        reduced_counts=(12 9)
        export CASE_ORDER="healthy nep"
    else
        reduced_counts=(9 12)
        export CASE_ORDER="nep healthy"
    fi
    for reduced_ga in "${reduced_counts[@]}"; do
        export NEP_REDUCED_NUM_MICROBATCHES="${reduced_ga}"
        export ROOT_DIR="${ROOT}/${SLURM_JOB_ID}/round${round}_ga12_${reduced_ga}"
        bash "${RUNNER}"
    done
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
