#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=2
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:18:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.random-gate
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export CODE_REPO=/home/darfeen/Megatron-LM/.worktrees/nep-placement-controlled-20260929
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe
export NEP_CONTROLLED_ROUTING_SEED=101
export CLEANUP_CASES='controlled_random_compact controlled_random_spread'
exec bash /home/darfeen/Megatron-LM/scripts/nonuniform/run_nep_cleanup_ab.sh
