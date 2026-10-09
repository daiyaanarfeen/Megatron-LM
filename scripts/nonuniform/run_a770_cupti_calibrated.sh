#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=51 SOFT_CLIQUE_TIMING_END=60
for SOFT_CLIQUE_MBS in 1 2; do
    export SOFT_CLIQUE_MBS
    if [[ "${SOFT_CLIQUE_MBS}" == 1 ]]; then
        export CASE_ORDER='nep healthy'
        variants='control profiled'
    else
        export CASE_ORDER='healthy nep'
        variants='profiled control'
    fi
    for variant in $variants; do
        export CUPTI_COLLECTION_DISABLED=0 SOFT_CLIQUE_TAG="mbs${SOFT_CLIQUE_MBS}"
        if [[ "$variant" == control ]]; then
            export CUPTI_COLLECTION_DISABLED=1 SOFT_CLIQUE_TAG="control_mbs${SOFT_CLIQUE_MBS}"
        fi
        bash scripts/nonuniform/run_lyris_smallest_soft_clique.sh
    done
done
