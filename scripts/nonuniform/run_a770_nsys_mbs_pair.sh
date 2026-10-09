#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:22:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES="${PROFILE_ACTIVITY_MODES:-nsys}"
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
for SOFT_CLIQUE_MBS in 1 2; do
    export SOFT_CLIQUE_MBS SOFT_CLIQUE_TAG="mbs${SOFT_CLIQUE_MBS}"
    if [[ "${SOFT_CLIQUE_MBS}" == 1 ]]; then
        export CASE_ORDER='nep healthy'
    else
        export CASE_ORDER='healthy nep'
    fi
    bash scripts/nonuniform/run_lyris_smallest_soft_clique.sh
done
