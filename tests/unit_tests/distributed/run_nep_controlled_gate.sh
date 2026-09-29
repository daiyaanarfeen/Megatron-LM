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
#SBATCH --job-name=coreai_comparch_sysarch-nep.control-gate
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export CODE_REPO=/home/darfeen/Megatron-LM/.worktrees/nep-placement-controlled-20260929
export GRAPH_ALLOW_DIRTY=0 GRAPH_SCOPES=moe
export CLEANUP_CASES='controlled_reference controlled_one controlled_one_repeat controlled_reference_repeat controlled_two controlled_spread_two controlled_reference_off controlled_one_off controlled_one_repeat_off controlled_reference_repeat_off'
exec bash /home/darfeen/Megatron-LM/scripts/nonuniform/run_nep_cleanup_ab.sh
