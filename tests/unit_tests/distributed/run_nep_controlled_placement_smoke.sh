#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.control-unit
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-placement-controlled-20260929
export CHECK_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_controlled_placement/$SLURM_JOB_ID
export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform"
export CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_LAUNCH_ORDER_IMPLICIT=0 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1
export CUPTI_COLLECTION_DISABLED=0
mkdir -p "$CHECK_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$CHECK_ROOT/allocation.txt"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(megatron/core/distributed/nonuniform_common.py megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nep_controlled_placement.py tests/unit_tests/distributed/nep_parameter_wave_smoke.py tests/unit_tests/distributed/nep_param_prefetch_smoke.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    git diff > "$CHECK_ROOT/implementation.patch"
    python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py tests/unit_tests/distributed/test_nep_controlled_placement.py > "$CHECK_ROOT/unit.log" 2>&1
    for policy in compact:ascending:1 compact:waves:1 compact:ascending:2 compact:waves:2 spread:ascending:1 spread:ascending:2; do
        IFS=: read -r NEP_ABLATION_PLACEMENT NEP_ABLATION_PARAM_ORDER NEP_ABLATION_PARAM_LANES <<< "$policy"
        export NEP_ABLATION_PLACEMENT NEP_ABLATION_PARAM_ORDER NEP_ABLATION_PARAM_LANES
        export CUPTI_OUTPUT_DIR="$CHECK_ROOT/$policy"
        timeout --signal=TERM --kill-after=10s 120 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_parameter_wave_smoke.py > "$CHECK_ROOT/$policy.log" 2>&1
    done
    for NEP_ABLATION_PARAM_LANES in 1 2; do
        export NEP_ABLATION_PARAM_LANES CUPTI_OUTPUT_DIR="$CHECK_ROOT/prefetch_$NEP_ABLATION_PARAM_LANES"
        timeout --signal=TERM --kill-after=10s 120 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_param_prefetch_smoke.py > "$CHECK_ROOT/prefetch_$NEP_ABLATION_PARAM_LANES.log" 2>&1
    done
    echo CONTROLLED_PLACEMENT_SMOKE_PASS
    '
