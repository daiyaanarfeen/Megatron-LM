#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:24:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti CUPTI_EVENTS_ONLY=1
export CUPTI_COLLECTION_DISABLED=0
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m CASE_ORDER=nep
export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=51 SOFT_CLIQUE_TIMING_END=60
candidate=/home/darfeen/Megatron-LM-nep-param-cpu-20260911
for SOFT_CLIQUE_MBS in 1 2; do
    export SOFT_CLIQUE_MBS
    variants='baseline candidate'
    if [[ "$SOFT_CLIQUE_MBS" == 2 ]]; then variants='candidate baseline'; fi
    for variant in $variants; do
        export PARAM_TRANSFER_CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
        if [[ "$variant" == candidate ]]; then export PARAM_TRANSFER_CODE_REPO="$candidate"; fi
        export SOFT_CLIQUE_TAG="${variant}_mbs${SOFT_CLIQUE_MBS}"
        bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    done
done
