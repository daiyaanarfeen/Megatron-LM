#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:08:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.compact-smoke
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export ROOT=/home/darfeen/Megatron-LM/.worktrees/nep-param-compact-20260917
export CUPTI_OUTPUT_DIR=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/compact_smoke_$SLURM_JOB_ID
export CUPTI_COLLECTION_DISABLED=0 CUDA_DEVICE_MAX_CONNECTIONS=32 TORCH_NCCL_BLOCKING_WAIT=0 NCCL_LAUNCH_ORDER_IMPLICIT=1 OMP_NUM_THREADS=1
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 --no-container-mount-home \
 --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
 bash -lc '
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$ROOT:/home/darfeen/Megatron-LM/scripts/nonuniform:${PYTHONPATH:-}"
uv run --no-project --active isort tests/unit_tests/distributed/nep_compact_params_smoke.py
python -m black --skip-string-normalization megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/nep_compact_params_smoke.py tests/unit_tests/distributed/test_nonuniform_ep.py
timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_compact_params_smoke.py
export CUPTI_OUTPUT_DIR="${CUPTI_OUTPUT_DIR}/prefetch"
timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_param_prefetch_smoke.py
timeout 120 python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py -k "nep_distopt_param_sync or nep_distopt_async_param_sync or nep_distopt_param_transfer"
'
