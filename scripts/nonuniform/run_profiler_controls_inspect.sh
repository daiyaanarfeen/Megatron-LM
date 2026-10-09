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
export PROFILER_INSPECT_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/profiler_controls_${SLURM_JOB_ID}
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
    --no-container-mount-home python /home/darfeen/Megatron-LM/scripts/nonuniform/inspect_profiler_controls.py
