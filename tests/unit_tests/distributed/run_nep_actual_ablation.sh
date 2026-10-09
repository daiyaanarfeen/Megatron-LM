#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.actual-gate
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-real-ablation-20261002
export CONTROLLED_CODE_REPO=$CODE_REPO
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe GRAPH_FULL_EP=16 GRAPH_REDUCED_EP=12 NCCL_PXN_DISABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=0 CLEANUP_CASE_TIMEOUT=15m
export CUDA_EVENT_POOL_SIZE=131072
# GA10 exercises the largest batch schedule; both transfer lanes are always preallocated.
export CLEANUP_CASES=${CLEANUP_CASES:-'controlled_actual_O_r0 controlled_actual_O_r0_off controlled_actual_H_r0 controlled_actual_H_r0_off controlled_actual_B_r0 controlled_actual_B_r0_off'}
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(tests/unit_tests/distributed/nep_actual_ablation.py tests/unit_tests/distributed/nep_controlled_probe.py tests/unit_tests/distributed/validate_nep_work_allocation.py tests/unit_tests/distributed/validate_nep_study.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    python -m py_compile "${files[@]}"
    '
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/$SLURM_JOB_ID
mkdir -p "$root"
cp "$base/scripts/nonuniform/run_nep_cleanup_ab.sh" "$root/run_nep_cleanup_ab.sh"
exec bash "$base/scripts/nonuniform/run_nep_cleanup_ab.sh"
