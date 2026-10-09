#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200,gb300
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.work-overload
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
# Largest reduced GA first; counterbalance configuration and instrumentation order.
export CLEANUP_CASES='controlled_work_O_r0 controlled_work_O_r0_off controlled_work_N1_r0 controlled_work_N1_r0_off controlled_work_H1_r0 controlled_work_H1_r0_off controlled_work_H1_r1_off controlled_work_H1_r1 controlled_work_N1_r1_off controlled_work_N1_r1 controlled_work_O_r1_off controlled_work_O_r1'
exec bash /home/darfeen/Megatron-LM/.worktrees/nep-work-allocation-20260930/tests/unit_tests/distributed/run_nep_work_allocation.sh
