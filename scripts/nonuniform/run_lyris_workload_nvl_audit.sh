#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail
# Submit with an explicit node list containing half the nodes in each NVL domain.
# The startup guard validates actual rank placement before any training begins.
export STRICT_NVL_PLACEMENT=1 SLURM_CPU_BIND=none
unset PRETRAIN_DIAGNOSTIC_ENTRYPOINT BENCH_CPUS_PER_TASK
unset HEALTHY_MICRO_BATCH_SIZE HEALTHY_NUM_MICROBATCHES NEP_FULL_MICRO_BATCH_SIZE NEP_REDUCED_MICRO_BATCH_SIZE
unset NEP_FULL_NUM_MICROBATCHES NEP_REDUCED_NUM_MICROBATCHES
unset CUDA_GRAPH_MODULES_OVERRIDE DISABLE_CUDA_GRAPHS DDP_NUM_BUCKETS_OVERRIDE EMPTY_UNUSED_MEMORY_LEVEL
export GROUP=${GROUP:-ep8}
export WORKLOAD_FILTER=${WORKLOAD_FILTER:-scaling_ladder/a500m_2b_moe_140b}
export CODE_REPO=${CODE_REPO:-/home/darfeen/Megatron-LM-nep-main-20260817}
export TRAIN_ITERS=24 PROFILE_STEP_START=10 PROFILE_STEP_END=14 TIMING_START=15 TIMING_END=24
export CASE_TIMEOUT=7m CASE_ORDER='healthy nep' BENCH_ROUTING_MODE=random
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/workload_nvl_audit/${SLURM_JOB_ID}
mkdir -p "${root}"
scontrol show job "${SLURM_JOB_ID}" > "${root}/allocation.txt"
scontrol show topology > "${root}/topology.txt"
git -C "${CODE_REPO}" rev-parse HEAD > "${root}/training_commit.txt"
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py" > "${root}/training_sha256.txt"
export ROOT_DIR=${root}/original
bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh

# Only a500 has an automatic, previously tested batch ladder in this first gate.
if [[ "${WORKLOAD_FILTER}" != scaling_ladder/a500m_2b_moe_140b ]]; then
    exit 0
fi
for setting in larger_dispatch below_proportional; do
    comparison=${ROOT_DIR}/${GROUP}/scaling_ladder__a500m_2b_moe_140b/comparison_${SLURM_JOB_ID}.json
    if python3 - "${comparison}" <<'PY'
import json,statistics,sys
d=json.load(open(sys.argv[1]));h,n=d['healthy'],d['nep']
mean_parity=h['mean_elapsed_ms']/n['mean_elapsed_ms']
median_parity=statistics.median(h['elapsed_ms'])/statistics.median(n['elapsed_ms'])
stable=all(x['std_elapsed_ms']/x['mean_elapsed_ms']<=.03 for x in (h,n))
print(f'Parity gate: mean={mean_parity:.4f} median={median_parity:.4f} stable={stable}',flush=True)
sys.exit(0 if mean_parity>=.95 and median_parity>=.95 and stable else 1)
PY
    then
        echo "Original/current configuration meets the 95% parity gate; no further tuning."
        break
    fi
    export ROOT_DIR=${root}/${setting}
    export HEALTHY_MICRO_BATCH_SIZE=3 NEP_FULL_MICRO_BATCH_SIZE=3 NEP_REDUCED_MICRO_BATCH_SIZE=3
    export HEALTHY_NUM_MICROBATCHES=3 NEP_FULL_NUM_MICROBATCHES=3 NEP_REDUCED_NUM_MICROBATCHES=3
    if [[ "${setting}" == below_proportional ]]; then
        export HEALTHY_NUM_MICROBATCHES=4 NEP_FULL_NUM_MICROBATCHES=4
    fi
    bash scripts/nonuniform/run_lyris_training_scripts_nep_ab.sh
done
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py"
