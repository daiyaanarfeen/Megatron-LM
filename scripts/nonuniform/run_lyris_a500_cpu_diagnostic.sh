#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --segment=4
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export GROUP=ep8 WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
export TRAIN_ITERS=24 PROFILE_STEP_START=16 PROFILE_STEP_END=22
export TIMING_START=17 TIMING_END=21 CASE_TIMEOUT=8m
export HEALTHY_NUM_MICROBATCHES=12 NEP_FULL_NUM_MICROBATCHES=12 NEP_REDUCED_NUM_MICROBATCHES=12
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
export ROOT_DIR=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_cpu_diagnostic/${SLURM_JOB_ID}
export PRETRAIN_DIAGNOSTIC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_cpu_diagnostic.py
export NEP_CPU_DIAG_OUTPUT=${ROOT_DIR}/cpu/${CASE_ORDER:-healthy_nep}
export NEP_CPU_DIAG_START=10 NEP_CPU_DIAG_END=22
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
