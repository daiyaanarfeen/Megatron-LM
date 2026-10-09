#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.placement-ab
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export CODE_REPO=${ABLATION_CODE_REPO:-/home/darfeen/Megatron-LM/.worktrees/nep-placement-ablation-20260928}
export GRAPH_ALLOW_DIRTY=0 GRAPH_SCOPES=${GRAPH_SCOPES:-moe}
export CLEANUP_CASES="${CLEANUP_CASES:-ablation_healthy ablation_current_waves ablation_current_ascending ablation_spread_waves ablation_spread_ascending ablation_current_repeat}"
exec bash /home/darfeen/Megatron-LM/scripts/nonuniform/run_nep_cleanup_ab.sh
