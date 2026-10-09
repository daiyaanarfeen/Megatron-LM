#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300-backfill
#SBATCH --nodes=4
#SBATCH --nodelist=theia[0028-0031]
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:28:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail
unset BENCH_CPUS_PER_TASK
export SLURM_CPU_BIND=none
export GROUP=ep8 WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
export TRAIN_ITERS=24 PROFILE_STEP_START=16 PROFILE_STEP_END=22
export TIMING_START=17 TIMING_END=21 CASE_TIMEOUT=7m
export HEALTHY_NUM_MICROBATCHES=12 NEP_FULL_NUM_MICROBATCHES=12
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
export PRETRAIN_DIAGNOSTIC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_cpu_diagnostic.py
export NEP_CPU_DIAG_START=10 NEP_CPU_DIAG_END=22 NEP_CPU_DIAG_SPLITS=1
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_routing_control/${SLURM_JOB_ID}
mkdir -p "${root}"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
read -r -a scenarios <<< "${ROUTING_SCENARIOS:-random_proportional uniform_proportional uniform_below}"
for scenario in "${scenarios[@]}"; do
    case "${scenario}" in
        random_proportional|uniform_proportional|uniform_below) ;;
        *) echo "Invalid routing scenario: ${scenario}" >&2; exit 2 ;;
    esac
    export ROOT_DIR=${root}/${scenario} NEP_REDUCED_NUM_MICROBATCHES=12
    export NEP_CPU_DIAG_OUTPUT=${ROOT_DIR}/cpu
    export NEP_CPU_DIAG_ROUTING=existing BENCH_ROUTING_MODE=random CASE_ORDER='healthy nep'
    if [[ "${scenario}" != random_proportional ]]; then
        export NEP_CPU_DIAG_ROUTING=uniform BENCH_ROUTING_MODE=uniform CASE_ORDER='nep healthy'
    fi
    if [[ "${scenario}" == uniform_below ]]; then
        export NEP_REDUCED_NUM_MICROBATCHES=9 CASE_ORDER='healthy nep'
    fi
    bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
