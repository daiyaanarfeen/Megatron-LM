#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.dest-scatter
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-param-destination-scatter-20260929
export CHECK_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_destination_scatter/$SLURM_JOB_ID
export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform"
export CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_LAUNCH_ORDER_IMPLICIT=0 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1
export CUPTI_COLLECTION_DISABLED=0
mkdir -p "$CHECK_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$CHECK_ROOT/allocation.txt"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py tests/unit_tests/distributed/nep_compact_params_smoke.py tests/unit_tests/distributed/nep_param_prefetch_smoke.py tests/unit_tests/distributed/nep_parameter_wave_smoke.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    git diff > "$CHECK_ROOT/implementation.patch"
    python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py > "$CHECK_ROOT/unit.log" 2>&1
    for smoke in nep_compact_params_smoke nep_param_prefetch_smoke nep_parameter_wave_smoke; do
        export CUPTI_OUTPUT_DIR="$CHECK_ROOT/$smoke"
        timeout --signal=TERM --kill-after=10s 150 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/$smoke.py > "$CHECK_ROOT/$smoke.log" 2>&1
    done
    echo DESTINATION_SCATTER_SMOKE_PASS
    '
