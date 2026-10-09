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
#SBATCH --job-name=coreai_comparch_sysarch-nep-sync-stress
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
runtime=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export PARAM_TRANSFER_CODE_REPO="$runtime"
export NEP_SYNC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_sync_stress.py
unset NEP_SYNC_REFERENCE NEP_SYNC_SHADOW
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 OMP_NUM_THREADS=1
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
export SOFT_CLIQUE_MBS=2 SOFT_CLIQUE_TRAIN_ITERS=60
export SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=40 SOFT_CLIQUE_TIMING_END=49
for tag in anchor_start expert_f3072 expert_f3072_seq2048 anchor_end; do
    export SOFT_CLIQUE_TAG="$tag" SYNC_WORKLOAD_VARIANT="$tag"
    case "$tag" in
        anchor_start) export SYNC_WORKLOAD_VARIANT=anchor CASE_ORDER='healthy nep' ;;
        expert_f3072) export CASE_ORDER='nep healthy' ;;
        expert_f3072_seq2048) export CASE_ORDER='healthy nep' ;;
        anchor_end) export SYNC_WORKLOAD_VARIANT=anchor CASE_ORDER='nep healthy' ;;
    esac
    echo "SYNC_STRESS_BEGIN $(date --iso-8601=seconds) $tag $CASE_ORDER"
    bash "$runtime/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    echo "SYNC_STRESS_DONE $(date --iso-8601=seconds) $tag"
done
