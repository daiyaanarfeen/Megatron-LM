#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
export SOFT_CLIQUE_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/${SLURM_JOB_ID}
if [[ -n "${SOFT_CLIQUE_TAG:-}" ]]; then
    export SOFT_CLIQUE_ROOT="${SOFT_CLIQUE_ROOT}/${SOFT_CLIQUE_TAG}"
fi
mkdir -p "${SOFT_CLIQUE_ROOT}"/{placement,transport}
scontrol show job "${SLURM_JOB_ID}" > "${SOFT_CLIQUE_ROOT}/allocation.txt"
python3 scripts/nonuniform/nep_nvl_placement.py --prepare "${SOFT_CLIQUE_ROOT}/domains.json"
mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export SOFT_CLIQUE_HOSTS="${hosts[*]}"
python3 - "${SOFT_CLIQUE_ROOT}/domains.json" "${hosts[@]}" <<'PY'
import json,sys
domains=json.load(open(sys.argv[1]))
assert len(sys.argv[2:]) in (4, 8)
hosts = sys.argv[2:]
# Both cases use contiguous ranks; NEP drops only the final replica's ranks.
for replica, replica_hosts in enumerate((hosts[:len(hosts)//2], hosts[len(hosts)//2:])):
    blocks = {domains[h] for h in replica_hosts}
    assert len(blocks) == 1, f'Replica {replica} spans NVL blocks: {blocks}'
    print('Replica:', replica, 'physical domain:', next(iter(blocks)), 'hosts:', replica_hosts)
PY
unset STRICT_NVL_PLACEMENT NEP_PLACEMENT_METADATA_DIR NEP_PLACEMENT_DOMAIN_FILE
unset PRETRAIN_DIAGNOSTIC_ENTRYPOINT BENCH_CPUS_PER_TASK
unset HEALTHY_MICRO_BATCH_SIZE HEALTHY_NUM_MICROBATCHES
unset NEP_FULL_MICRO_BATCH_SIZE NEP_REDUCED_MICRO_BATCH_SIZE NEP_FULL_NUM_MICROBATCHES NEP_REDUCED_NUM_MICROBATCHES
export HEALTHY_NUM_MICROBATCHES=2 NEP_FULL_NUM_MICROBATCHES=2 NEP_REDUCED_NUM_MICROBATCHES=2
if [[ -n "${SOFT_CLIQUE_MBS:-}" ]]; then
    [[ "${SOFT_CLIQUE_MBS}" =~ ^[1-9][0-9]*$ ]]
    export HEALTHY_MICRO_BATCH_SIZE="${SOFT_CLIQUE_MBS}"
    export NEP_FULL_MICRO_BATCH_SIZE="${SOFT_CLIQUE_MBS}"
    export NEP_REDUCED_MICRO_BATCH_SIZE="${SOFT_CLIQUE_MBS}"
fi
unset CUDA_GRAPH_MODULES_OVERRIDE DISABLE_CUDA_GRAPHS DDP_NUM_BUCKETS_OVERRIDE EMPTY_UNUSED_MEMORY_LEVEL
unset NCCL_MNNVL_ENABLE NCCL_MNNVL_UUID NCCL_P2P_DISABLE
export SLURM_CPU_BIND=none GROUP="${GROUP:-ep8}" WORKLOAD_FILTER="${WORKLOAD_FILTER:-scaling_ladder/a315m_1b_moe_88b}"
export CASE_ORDER="${CASE_ORDER:-healthy}" TRAIN_ITERS="${SOFT_CLIQUE_TRAIN_ITERS:-24}" PROFILE_STEP_START="${SOFT_CLIQUE_PROFILE_START:-10}" PROFILE_STEP_END="${SOFT_CLIQUE_PROFILE_END:-14}" TIMING_START="${SOFT_CLIQUE_TIMING_START:-15}" TIMING_END="${SOFT_CLIQUE_TIMING_END:-24}"
export CASE_TIMEOUT="${CASE_TIMEOUT:-7m}" BENCH_ROUTING_MODE=random ROOT_DIR="${SOFT_CLIQUE_ROOT}"
export RANK_LAUNCHER=/home/darfeen/Megatron-LM/scripts/nonuniform/run_lyris_soft_clique_rank.sh
export CODE_REPO="${PARAM_TRANSFER_CODE_REPO:?}"
git -C "${CODE_REPO}" rev-parse HEAD > "${SOFT_CLIQUE_ROOT}/training_commit.txt"
sha256sum "${CODE_REPO}/megatron/core/distributed/nonuniform_ep.py" > "${SOFT_CLIQUE_ROOT}/training_sha256.txt"
if [[ "${SOFT_CLIQUE_PROFILE_SWEEP:-0}" == 1 ]]; then
    export PRETRAIN_DIAGNOSTIC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_profile_activity_probe.py
    if [[ -n "${NEP_SYNC_ENTRYPOINT:-}" ]]; then
        export PRETRAIN_DIAGNOSTIC_ENTRYPOINT="${NEP_SYNC_ENTRYPOINT}"
    fi
    base_root="${SOFT_CLIQUE_ROOT}"
    read -r -a profile_modes <<< "${PROFILE_ACTIVITY_MODES:-cpu_cuda cuda}"
    for PROFILE_ACTIVITY_MODE in "${profile_modes[@]}"; do
        [[ "${PROFILE_ACTIVITY_MODE}" == cpu_cuda || "${PROFILE_ACTIVITY_MODE}" == cuda || "${PROFILE_ACTIVITY_MODE}" == nsys || "${PROFILE_ACTIVITY_MODE}" == cupti ]]
        export PROFILE_ACTIVITY_MODE
        export SOFT_CLIQUE_ROOT="${base_root}/${PROFILE_ACTIVITY_MODE}"
        export ROOT_DIR="${SOFT_CLIQUE_ROOT}"
        mkdir -p "${SOFT_CLIQUE_ROOT}"/{placement,transport}
        cp "${base_root}/allocation.txt" "${SOFT_CLIQUE_ROOT}/allocation.txt"
        bash /home/darfeen/Megatron-LM/experiments/nep_cleanup_20260921/nep_param_transfer_cases.sh
    done
else
    bash /home/darfeen/Megatron-LM/experiments/nep_cleanup_20260921/nep_param_transfer_cases.sh
fi
