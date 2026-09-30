#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:08:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.random-unit
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-placement-controlled-20260929
export GRAPH_OUTPUT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_randomized_controls/$SLURM_JOB_ID
export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform" RANK=0 NEP_CONTROLLED_ROUTING_SEED=101
export CUDA_DEVICE_MAX_CONNECTIONS=32
mkdir -p "$GRAPH_OUTPUT"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(tests/unit_tests/distributed/nep_balanced_routing.py tests/unit_tests/distributed/nep_randomized_controls.py tests/unit_tests/distributed/nep_controlled_probe.py tests/unit_tests/distributed/test_nep_balanced_routing_cpu.py tests/unit_tests/distributed/nep_randomized_routing_smoke.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    python tests/unit_tests/distributed/test_nep_balanced_routing_cpu.py
    timeout --signal=TERM --kill-after=10s 180 python tests/unit_tests/distributed/nep_randomized_routing_smoke.py
    '
