#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:06:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export CUPTI_OUTPUT_DIR=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/cuda_event_smoke_${SLURM_JOB_ID}
export PYTHONPATH=/home/darfeen/Megatron-LM-nep-main-20260817:/home/darfeen/Megatron-LM/scripts/nonuniform
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,/home/darfeen/Megatron-LM-nep-main-20260817:/home/darfeen/Megatron-LM-nep-main-20260817,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
    --no-container-mount-home python /home/darfeen/Megatron-LM/scripts/nonuniform/cuda_event_capture_smoke.py
