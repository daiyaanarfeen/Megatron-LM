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
export GROUP=ep8 WORKLOAD_FILTER=scaling_ladder/a500m_2b_moe_140b
export TRAIN_ITERS=24 PROFILE_STEP_START=16 PROFILE_STEP_END=22
export TIMING_START=17 TIMING_END=21 CASE_TIMEOUT=7m
export HEALTHY_NUM_MICROBATCHES=12 NEP_FULL_NUM_MICROBATCHES=12
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
export PRETRAIN_DIAGNOSTIC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_cpu_diagnostic.py
export NEP_CPU_DIAG_START=10 NEP_CPU_DIAG_END=22
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a500_cpu_masks/${SLURM_JOB_ID}
mkdir -p "${root}"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
mask_list=$(python3 -c 'print(",".join(hex(((1 << 36)-1) << (36*i)) for i in range(4)))')
for scenario in unbound_proportional bound_proportional bound_below; do
    export ROOT_DIR=${root}/${scenario} NEP_REDUCED_NUM_MICROBATCHES=12
    export NEP_CPU_DIAG_OUTPUT=${ROOT_DIR}/cpu
    export SLURM_CPU_BIND=none CASE_ORDER='healthy nep'
    if [[ "${scenario}" != unbound_proportional ]]; then
        export SLURM_CPU_BIND="verbose,mask_cpu:${mask_list}" CASE_ORDER='nep healthy'
        # Explicit masks use local GPU IDs, including nodes with only three ranks.
        srun --overlap --nodes=4 --ntasks=14 --ntasks-per-node=4 --mpi=none \
            python3 -c 'import os; r=int(os.environ["SLURM_PROCID"]); n=int(os.environ["SLURM_NODEID"]); l=int(os.environ["SLURM_LOCALID"]); expected=[(i,j) for i,c in enumerate((4,4,3,3)) for j in range(c)]; assert (n,l)==expected[r], (r,n,l); assert set(os.sched_getaffinity(0))==set(range(l*36,(l+1)*36)); print(r,os.uname().nodename,l,sorted(os.sched_getaffinity(0)),flush=True)'
    fi
    if [[ "${scenario}" == bound_below ]]; then
        export NEP_REDUCED_NUM_MICROBATCHES=9 CASE_ORDER='healthy nep'
    fi
    bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
