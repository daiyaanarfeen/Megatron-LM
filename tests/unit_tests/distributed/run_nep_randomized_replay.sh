#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.ready-replay
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-placement-controlled-20260929
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe NEP_CONTROLLED_REPLAY=1
export CLEANUP_CASES='controlled_replay_A controlled_replay_E controlled_replay_F controlled_replay_C controlled_replay_B controlled_replay_D'
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(tests/unit_tests/distributed/nep_randomized_controls.py tests/unit_tests/distributed/nep_all_ready_replay.py tests/unit_tests/distributed/validate_nep_randomized_controls.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    '
exec bash "$base/scripts/nonuniform/run_nep_cleanup_ab.sh"
