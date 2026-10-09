#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300-backfill
#SBATCH --nodes=8
#SBATCH --nodelist=theia[0176-0178,0180,0187,0189,0196-0197]
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail
unset PRETRAIN_DIAGNOSTIC_ENTRYPOINT BENCH_CPUS_PER_TASK
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t
export TRAIN_ITERS=10 PROFILE_STEP_START=5 PROFILE_STEP_END=7
export TIMING_START=8 TIMING_END=10 CASE_ORDER='healthy nep' CASE_TIMEOUT=12m
export HEALTHY_MICRO_BATCH_SIZE=2 NEP_FULL_MICRO_BATCH_SIZE=2 NEP_REDUCED_MICRO_BATCH_SIZE=2
export CODE_REPO=/home/darfeen/Megatron-LM-nep-main-20260817
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/a8b_fixed_placement/${SLURM_JOB_ID}
mkdir -p "${root}"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
scontrol show topology > "${root}/slurm-topology.txt"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --output="${root}/gpu-topology-%N.txt" nvidia-smi topo -m
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
for ga in 20 4; do
    export ROOT_DIR=${root}/ga${ga}
    export HEALTHY_NUM_MICROBATCHES=${ga} NEP_FULL_NUM_MICROBATCHES=${ga} NEP_REDUCED_NUM_MICROBATCHES=${ga}
    bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
