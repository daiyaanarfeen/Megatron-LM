#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:28:00
#SBATCH --segment=4
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export GROUP=ep8 WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
export TRAIN_ITERS=24 PROFILE_STEP_START=16 PROFILE_STEP_END=22
export TIMING_START=17 TIMING_END=21 CASE_TIMEOUT=7m
export HEALTHY_NUM_MICROBATCHES=12 NEP_FULL_NUM_MICROBATCHES=12
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
export PRETRAIN_DIAGNOSTIC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_cpu_diagnostic.py
export NEP_CPU_DIAG_START=10 NEP_CPU_DIAG_END=22
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_cpu_placement/${SLURM_JOB_ID}
mkdir -p "${root}"
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --output="${root}/cpu-topology-%N.json" \
    lscpu --json --extended=CPU,NODE,SOCKET,CORE
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
for scenario in unbound_proportional bound_proportional bound_below; do
    export ROOT_DIR=${root}/${scenario}
    export NEP_CPU_DIAG_OUTPUT=${ROOT_DIR}/cpu
    export NEP_REDUCED_NUM_MICROBATCHES=12
    if [[ "${scenario}" == unbound_proportional ]]; then
        unset BENCH_CPUS_PER_TASK
        export CASE_ORDER='healthy nep'
    else
        export BENCH_CPUS_PER_TASK=36
        export CASE_ORDER='nep healthy'
    fi
    if [[ "${scenario}" == bound_below ]]; then
        export NEP_REDUCED_NUM_MICROBATCHES=9
        export CASE_ORDER='healthy nep'
    fi
    bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
