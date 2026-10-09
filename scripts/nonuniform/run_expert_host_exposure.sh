#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --segment=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --job-name=coreai_comparch_sysarch-nep.host-cause
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/expert_host_exposure/${SLURM_JOB_ID}
mkdir -p "$root"
scontrol show job "$SLURM_JOB_ID" > "$root/allocation.txt"
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTHONPATH="$base/.worktrees/nep-slot-cache-20260917"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 \
    --container-name=nep_nemo_26_06 --no-container-mount-home \
    --container-mounts="$base:$base,/lustre:/lustre" \
    bash -lc "set -e; cd '$base'; uv run --no-project --active isort scripts/nonuniform/probe_expert_host_exposure.py; nvidia-smi > '$root/gpus.txt'; timeout --signal=TERM --kill-after=30s 7m python scripts/nonuniform/probe_expert_host_exposure.py --output '$root'"
