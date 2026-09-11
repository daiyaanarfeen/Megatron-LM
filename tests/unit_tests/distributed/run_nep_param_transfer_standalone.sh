#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:06:00
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ROOT=/home/darfeen/Megatron-LM-nep-param-cpu-20260911
BASE=/home/darfeen/Megatron-LM-nep-main-20260817
export ROOT BASE
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
  --container-mounts=$ROOT:$ROOT,$BASE:$BASE,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
  --no-container-mount-home bash -c '
    set -euo pipefail
    cd "$ROOT"
    uv run --no-project --active isort tests/unit_tests/distributed/nep_param_transfer_standalone.py
    python -m black --skip-string-normalization megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/nep_param_transfer_standalone.py
    timeout --signal=TERM --kill-after=10s 180 python tests/unit_tests/distributed/nep_param_transfer_standalone.py "$BASE" "$ROOT"
  '
