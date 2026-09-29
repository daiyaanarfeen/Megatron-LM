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
#SBATCH --job-name=coreai_comparch_sysarch-nep.dest-scatter-ab
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export CODE_REPO=/home/darfeen/Megatron-LM/.worktrees/nep-param-destination-scatter-20260929
export ABLATION_CODE_REPO=$CODE_REPO
export GRAPH_ALLOW_DIRTY=0 GRAPH_SCOPES=moe
# Only the destination-specific parameter redistribution differs between NEP cases.
export CLEANUP_CASES='param_waves_margin ablation_current_waves ablation_healthy'
exec bash /home/darfeen/Megatron-LM/scripts/nonuniform/run_nep_cleanup_ab.sh
