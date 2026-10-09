#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:06:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.agrs-smoke
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
runtime=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export CUPTI_OUTPUT_DIR=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/collective_smoke_${SLURM_JOB_ID}
export PYTHONPATH="${runtime}:/home/darfeen/Megatron-LM/scripts/nonuniform"
export TORCH_NCCL_BLOCKING_WAIT=0 CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_LAUNCH_ORDER_IMPLICIT=1
export OMP_NUM_THREADS=1
timeout --foreground --signal=TERM --kill-after=30s 4m \
    srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
    --no-container-mount-home \
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
    /home/darfeen/Megatron-LM/scripts/nonuniform/cuda_collective_capture_smoke.py
