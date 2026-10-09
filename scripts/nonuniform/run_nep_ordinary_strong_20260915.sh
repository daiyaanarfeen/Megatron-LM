#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM/experiments/nep_ordinary_strong_20260915
#SBATCH --job-name=coreai_comparch_sysarch-nep.ordinary
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
export OMP_NUM_THREADS=1 SOFT_CLIQUE_MBS=2
for tag in anchor_start seq2048 seq1024 anchor_end; do
    export SOFT_CLIQUE_TAG="$tag"
    case "$tag" in
        anchor_start) export SENSITIVITY_VARIANT=anchor CASE_ORDER='healthy nep' ;;
        seq2048) export SENSITIVITY_VARIANT=seq2048 CASE_ORDER='nep healthy' ;;
        seq1024) export SENSITIVITY_VARIANT=seq1024 CASE_ORDER='healthy nep' ;;
        anchor_end) export SENSITIVITY_VARIANT=anchor CASE_ORDER='nep healthy' ;;
    esac
    echo "ORDINARY_BEGIN $(date --iso-8601=seconds) $tag $CASE_ORDER"
    bash scripts/nonuniform/run_lyris_smallest_soft_clique.sh
    echo "ORDINARY_COMPLETE $(date --iso-8601=seconds) $tag"
done
