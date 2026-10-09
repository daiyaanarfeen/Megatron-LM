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
#SBATCH --job-name=coreai_comparch_sysarch-nep.a8b-current
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-param-prefetch-20260915
reference=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE="$reference"
unset NEP_SYNC_SHADOW NEP_SYNC_ENTRYPOINT SOFT_CLIQUE_MBS SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export SOFT_CLIQUE_MBS="${A8B_MBS:-1}"
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=65536 OMP_NUM_THREADS=1
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT="${A8B_CASE_TIMEOUT:-13m}"
export SOFT_CLIQUE_TRAIN_ITERS=45 SOFT_CLIQUE_PROFILE_START=25 SOFT_CLIQUE_PROFILE_END=35
export SOFT_CLIQUE_TIMING_START=25 SOFT_CLIQUE_TIMING_END=34
export SOFT_CLIQUE_TAG="${A8B_TAG:-current}"
read -r -a cases <<< "${A8B_CASE_ORDER:-healthy nep}"
for case_name in "${cases[@]}"; do
    [[ "$case_name" == healthy || "$case_name" == nep ]]
    export CASE_ORDER="$case_name" PARAM_TRANSFER_CODE_REPO="$candidate"
    expected_sha=91f23f619d898f4678dcd850c1f71a328c11f0ca663a60df046500248d6788a7
    if [[ "$case_name" == healthy ]]; then
        export PARAM_TRANSFER_CODE_REPO="$reference"
        expected_sha=653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7
    fi
    actual_sha=$(sha256sum "$PARAM_TRANSFER_CODE_REPO/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)
    [[ "$actual_sha" == "$expected_sha" ]]
    echo "A8B_CURRENT_BEGIN $(date --iso-8601=seconds) $case_name $actual_sha"
    bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    echo "A8B_CURRENT_DONE $(date --iso-8601=seconds) $case_name"
done
