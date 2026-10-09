#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM-nep-sensitivity-accepted-20260914/sensitivity_harness
#SBATCH --job-name=coreai_comparch_sysarch-nep.knobs
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export BENCH_REPO="$PWD"
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=40 SOFT_CLIQUE_TIMING_END=49
export OMP_NUM_THREADS=1
matrix="${1:?Specify matrix A or B}"
case "$matrix" in
    A) tags='anchor_start mbs1 anchor_end' ;;
    B) tags='anchor_start seq4096 topk2 anchor_end' ;;
    *) exit 2 ;;
esac
for tag in $tags; do
    export SOFT_CLIQUE_MBS=2
    export SOFT_CLIQUE_TAG="$tag"
    case "$tag" in
        anchor_start) export SENSITIVITY_VARIANT=anchor CASE_ORDER='healthy nep' ;;
        mbs1) export SENSITIVITY_VARIANT=mbs1 SOFT_CLIQUE_MBS=1 CASE_ORDER='nep healthy' ;;
        seq4096) export SENSITIVITY_VARIANT=seq4096 CASE_ORDER='nep healthy' ;;
        topk2) export SENSITIVITY_VARIANT=topk2 CASE_ORDER='healthy nep' ;;
        anchor_end) export SENSITIVITY_VARIANT=anchor CASE_ORDER='nep healthy' ;;
    esac
    if [[ "$matrix" == A && "$tag" == anchor_end ]]; then
        export CASE_ORDER='healthy nep'
    fi
    echo "SENSITIVITY_BEGIN $(date --iso-8601=seconds) matrix=$matrix $tag $CASE_ORDER"
    bash scripts/nonuniform/run_lyris_smallest_soft_clique.sh
    echo "SENSITIVITY_COMPLETE $(date --iso-8601=seconds) $tag"
done
