#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep-sync
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE=/home/darfeen/Megatron-LM-nep-main-20260817
export SYNC_BASE_CODE_REPO="${SYNC_BASE_CODE_REPO:-$NEP_SYNC_REFERENCE}"
export NEP_SYNC_ENTRYPOINT="$candidate/tests/unit_tests/distributed/nep_sync_shadow_entrypoint.py"
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti CUPTI_EVENTS_ONLY=1
export CUPTI_COLLECTION_DISABLED=0 OMP_NUM_THREADS=1 CASE_ORDER=nep
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
# Build the generated helper in the isolated checkout, not in the trusted tree.
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 \
    --container-name=nep_nemo_26_06 --no-container-mount-home \
    --container-mounts="/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,$SYNC_BASE_CODE_REPO:$SYNC_BASE_CODE_REPO" \
    bash -c "set -e; cd '$candidate'; make -C megatron/core/datasets; make -C '$SYNC_BASE_CODE_REPO/megatron/core/datasets'; uv run --no-project --active isort tests/unit_tests/distributed/nep_sync_shadow_entrypoint.py tests/unit_tests/distributed/nep_param_transfer_standalone.py tests/unit_tests/distributed/nep_grad_stage_standalone.py; python -m black --skip-string-normalization megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/nep_sync_shadow_entrypoint.py tests/unit_tests/distributed/nep_param_transfer_standalone.py tests/unit_tests/distributed/nep_grad_stage_standalone.py; timeout 120 python tests/unit_tests/distributed/nep_param_transfer_standalone.py '$SYNC_BASE_CODE_REPO' '$candidate'; timeout 120 python tests/unit_tests/distributed/nep_grad_stage_standalone.py '$SYNC_BASE_CODE_REPO' '$candidate'"
for tag in ${SYNC_CASES:-shadow candidate_mbs1 original_mbs1 original_mbs2 candidate_mbs2}; do
    export CASE_ORDER=nep
    export SOFT_CLIQUE_TAG="$tag" NEP_SYNC_SHADOW=0 SOFT_CLIQUE_MBS=1
    export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
    export SOFT_CLIQUE_TIMING_START=40 SOFT_CLIQUE_TIMING_END=49
    export PARAM_TRANSFER_CODE_REPO="$candidate"
    if [[ "$tag" == original* ]]; then export PARAM_TRANSFER_CODE_REPO="$SYNC_BASE_CODE_REPO"; fi
    if [[ "$tag" == trusted* || "$tag" == healthy* ]]; then export PARAM_TRANSFER_CODE_REPO="$NEP_SYNC_REFERENCE"; fi
    if [[ "$tag" == healthy* ]]; then export CASE_ORDER=healthy; fi
    if [[ "$tag" == *mbs2 ]]; then export SOFT_CLIQUE_MBS=2; fi
    if [[ "$tag" == shadow ]]; then
        export NEP_SYNC_SHADOW=1 SOFT_CLIQUE_TRAIN_ITERS=8
        export SOFT_CLIQUE_PROFILE_START=5 SOFT_CLIQUE_PROFILE_END=7
        export SOFT_CLIQUE_TIMING_START=7 SOFT_CLIQUE_TIMING_END=8
    fi
    echo "NEP_SYNC_CASE_BEGIN $(date --iso-8601=seconds) $tag $PARAM_TRANSFER_CODE_REPO"
    bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    echo "NEP_SYNC_CASE_DONE $(date --iso-8601=seconds) $tag"
done
