#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.a8b-gb200-ab
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-param-compact-20260917
before=/home/darfeen/Megatron-LM/.worktrees/nep-param-prefetch-20260915
reference=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE="$reference"
unset NEP_SYNC_SHADOW SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export NEP_SYNC_ENTRYPOINT="${A8B_DIAGNOSTIC_ENTRYPOINT:-/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_a8b_gb200_depth.py}"
export SOFT_CLIQUE_MBS=2 SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=65536 OMP_NUM_THREADS=1
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT=12m
export SOFT_CLIQUE_TRAIN_ITERS=45 SOFT_CLIQUE_PROFILE_START=25 SOFT_CLIQUE_PROFILE_END=35
export SOFT_CLIQUE_TIMING_START=25 SOFT_CLIQUE_TIMING_END=34
# Build before distributed startup; do not rely on first-use helper compilation.
srun --nodes=1 --ntasks=1 --mpi=none \
    --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM \
    bash -lc "set -e; make -C '$candidate/megatron/core/datasets'; uv run --no-project --active isort '$NEP_SYNC_ENTRYPOINT'; python -m py_compile '$NEP_SYNC_ENTRYPOINT'"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --mpi=none \
    --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM \
    bash -lc "cd '$candidate'; python -c 'from megatron.core.datasets import helpers_cpp; print(helpers_cpp.__file__)'"
read -r -a variants <<< "${A8B_VARIANTS:-before after healthy}"
for variant in "${variants[@]}"; do
    export CASE_ORDER=nep SOFT_CLIQUE_TAG="gb200_${variant}"
    case "$variant" in
        before)
            export PARAM_TRANSFER_CODE_REPO="$before"
            expected_sha=91f23f619d898f4678dcd850c1f71a328c11f0ca663a60df046500248d6788a7
            ;;
        after)
            export PARAM_TRANSFER_CODE_REPO="$candidate"
            expected_sha=677332c72a1242ccd61334901c60b37c34500f021ecc96a527ffeec09b28966d
            ;;
        healthy)
            export CASE_ORDER=healthy PARAM_TRANSFER_CODE_REPO="$reference"
            expected_sha=653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7
            ;;
        *) exit 2 ;;
    esac
    actual_sha=$(sha256sum "$PARAM_TRANSFER_CODE_REPO/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)
    [[ "$actual_sha" == "$expected_sha" ]]
    export A8B_DEPTH_EXPECTED_SHA="$expected_sha"
    echo "A8B_GB200_BEGIN $(date --iso-8601=seconds) $variant $actual_sha"
    bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    echo "A8B_GB200_DONE $(date --iso-8601=seconds) $variant"
done
