#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.work-allocation
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-work-allocation-20260930
export CONTROLLED_CODE_REPO=$CODE_REPO
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe GRAPH_FULL_EP=16 GRAPH_REDUCED_EP=12 NCCL_PXN_DISABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=0
# The memory gate executes both MBS2 cases before any MBS1 study case.
export CLEANUP_CASES=${CLEANUP_CASES:-'controlled_work_N2_r0 controlled_work_H2_r0'}
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(tests/unit_tests/distributed/nep_randomized_controls.py tests/unit_tests/distributed/nep_controlled_probe.py tests/unit_tests/distributed/validate_nep_work_allocation.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    python -m py_compile "${files[@]}"
    uv run --no-project --active isort /home/darfeen/Megatron-LM/scripts/nonuniform/analyze_nep_work_allocation.py
    '
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/$SLURM_JOB_ID
mkdir -p "$root"
cp "$base/scripts/nonuniform/run_nep_cleanup_ab.sh" "$root/run_nep_cleanup_ab.sh"
exec bash "$base/scripts/nonuniform/run_nep_cleanup_ab.sh"
