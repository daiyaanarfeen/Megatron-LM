#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.a8b-compact
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-param-compact-20260917
reference=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE="$reference"
unset NEP_SYNC_SHADOW NEP_SYNC_ENTRYPOINT SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export SOFT_CLIQUE_MBS=2 SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=65536 OMP_NUM_THREADS=1
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT=18m
export SOFT_CLIQUE_TRAIN_ITERS=45 SOFT_CLIQUE_PROFILE_START=25 SOFT_CLIQUE_PROFILE_END=35
export SOFT_CLIQUE_TIMING_START=25 SOFT_CLIQUE_TIMING_END=34 SOFT_CLIQUE_TAG=mbs2
for case_name in nep healthy; do
    export CASE_ORDER="$case_name" PARAM_TRANSFER_CODE_REPO="$candidate"
    expected_sha="${COMPACT_EXPECTED_SHA:?Pin tested implementation at submission}"
    if [[ "$case_name" == healthy ]]; then
        export PARAM_TRANSFER_CODE_REPO="$reference"
        expected_sha=653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7
    fi
    actual_sha=$(sha256sum "$PARAM_TRANSFER_CODE_REPO/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)
    [[ "$actual_sha" == "$expected_sha" ]]
    echo "A8B_COMPACT_BEGIN $(date --iso-8601=seconds) $case_name $actual_sha"
    bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    echo "A8B_COMPACT_DONE $(date --iso-8601=seconds) $case_name"
done
