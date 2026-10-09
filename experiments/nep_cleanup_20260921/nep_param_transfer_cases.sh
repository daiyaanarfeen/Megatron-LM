#!/bin/bash

# Same-allocation healthy/NEP comparisons for one EP scale group from
# examples/training_scripts. Resource shape and GROUP are supplied by sbatch.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail

: "${GROUP:?Set GROUP to ep8, ep16, ep32, or ep64}"

BENCH_REPO="${BENCH_REPO:-/home/darfeen/Megatron-LM}"
CODE_REPO="${CODE_REPO:-/home/darfeen/Megatron-LM-nep-main-20260817}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/coreai_comparch_sysarch/darfeen}"
MATRIX="${BENCH_REPO}/scripts/nonuniform/training_scripts_nep_matrix.py"
RANK_LAUNCHER="${RANK_LAUNCHER:-${BENCH_REPO}/scripts/nonuniform/run_lyris_training_scripts_nep_rank.sh}"
BASE_RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh"
if [[ "${PROFILE_ACTIVITY_MODE:-}" == nsys ]]; then
    BASE_RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_nsys_rank.sh"
fi
ENTRYPOINT="${CODE_REPO}/examples/nonuniform/pretrain_hybrid_nonuniform.py"
ENTRYPOINT="${PRETRAIN_DIAGNOSTIC_ENTRYPOINT:-${ENTRYPOINT}}"
ROOT_DIR="${ROOT_DIR:-${ARTIFACT_ROOT}/slurm_runs/training_scripts_nep_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
TIMING_START="${TIMING_START:-8}"
TIMING_END="${TIMING_END:-${TRAIN_ITERS}}"
CASE_ORDER="${CASE_ORDER:-healthy nep}"
if [[ "${CASE_ORDER}" != "healthy nep" && "${CASE_ORDER}" != "nep healthy" && "${CASE_ORDER}" != healthy && "${CASE_ORDER}" != nep ]]; then
    echo "CASE_ORDER must be 'healthy nep', 'nep healthy', or 'healthy'" >&2
    exit 2
fi
read -r -a case_order <<< "${CASE_ORDER}"
CASE_TIMEOUT="${CASE_TIMEOUT:-35m}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-}"
cpu_binding_args=()
if [[ -n "${BENCH_CPUS_PER_TASK:-}" ]]; then
    [[ "${BENCH_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || exit 2
    cpu_binding_args=(--cpus-per-task="${BENCH_CPUS_PER_TASK}" --cpu-bind=verbose,cores --distribution=block:block)
fi

mkdir -p "${ROOT_DIR}/${GROUP}" "${ARTIFACT_ROOT}/slurm_runs/lyris"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
FIRST_NODE="${allocated_nodes[0]}"

reference_mount=""
if [[ -n "${NEP_SYNC_REFERENCE:-}" && "${NEP_SYNC_REFERENCE}" != "${CODE_REPO}" ]]; then
    reference_mount=",${NEP_SYNC_REFERENCE}:${NEP_SYNC_REFERENCE}"
fi
container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${BENCH_REPO}:${BENCH_REPO},${CODE_REPO}:${CODE_REPO},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}${reference_mount}"
    --container-workdir="${CODE_REPO}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

# Populate node-local Enroot caches once before the timed cases.
srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc 'python -c "import torch, transformer_engine, mamba_ssm, causal_conv1d; print(torch.__version__)"'

srun --nodes=1 --nodelist="${FIRST_NODE}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${CODE_REPO}'
python -m py_compile \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_hybrid_nonuniform.py
"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT=${NEP_LAUNCH_ORDER_IMPLICIT:-1}
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=${NEP_TEST_BWD_SM_MARGIN:-16}
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MNNVL=1

# Performance-retaining NEP configuration validated with Flex and DistOpt.
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0
export MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0
export MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1
export MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1
export MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0
export MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0
export MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1
export MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0
export MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1
export MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW=1
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter

if [[ -n "${WORKLOAD_FILTER}" ]]; then
    IFS=':' read -r -a workloads <<< "${WORKLOAD_FILTER}"
else
    mapfile -t workloads < <(python3 "${MATRIX}" --repo "${BENCH_REPO}" list --group "${GROUP}")
fi
if ((${#workloads[@]} == 0)); then
    echo "No workloads selected for ${GROUP}" >&2
    exit 2
fi

failures=()
case_index=0
for workload in "${workloads[@]}"; do
    slug="${workload//\//__}"
    healthy_replica_micro_batch_sizes=""
    healthy_replica_num_microbatches=""
    nep_replica_micro_batch_sizes=""
    nep_replica_num_microbatches=""
    echo "[training-scripts-nep] $(date --iso-8601=seconds) workload=${workload}"
    for case_name in "${case_order[@]}"; do
        IFS=$'\t' read -r tp cp etp _ mbs accum topology mode world_size run_nodes gbs ddp_buckets segment_nodes source_hybrid_ep_domain < <(
            python3 "${MATRIX}" --repo "${BENCH_REPO}" case-fields "${workload}" "${case_name}"
        )
        if ((run_nodes > ${#allocated_nodes[@]})); then
            echo "${workload}/${case_name}: needs ${run_nodes} nodes, allocation has ${#allocated_nodes[@]}" >&2
            exit 2
        fi
        job_segment=$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*SegmentSize=\([0-9][0-9]*\).*/\1/p')
        if [[ "${STRICT_NVL_PLACEMENT:-0}" == 1 ]]; then
            read -r full_units _ <<< "${topology}"
            segment_nodes=$((full_units * tp * cp / 4))
        fi
        if [[ -n "${SOFT_CLIQUE_ROOT:-}" ]]; then
            segment_nodes=4
        fi
        if [[ -n "${job_segment}" && "${job_segment}" != "${segment_nodes}" ]]; then
            echo "${workload}/${case_name}: expected segment=${segment_nodes}, got ${job_segment}" >&2
            exit 2
        fi
        run_dir="${ROOT_DIR}/${GROUP}/${slug}/${case_name}/${SLURM_JOB_ID}"
        mkdir -p "${run_dir}/tensorboard" "${run_dir}/torch_profile"
        if [[ "${PROFILE_ACTIVITY_MODE:-}" == nsys ]]; then
            export NSYS_OUTPUT_DIR="${run_dir}/nsys_profile"
        fi
        if [[ "${PROFILE_ACTIVITY_MODE:-}" == cupti ]]; then
            export CUPTI_OUTPUT_DIR="${run_dir}/cupti_profile"
        fi
        if [[ "${STRICT_NVL_PLACEMENT:-0}" == 1 ]]; then
            export NEP_PLACEMENT_METADATA_DIR="${run_dir}/placement"
            export NEP_PLACEMENT_DOMAIN_FILE="${run_dir}/domains.json"
            python3 "${BENCH_REPO}/scripts/nonuniform/nep_nvl_placement.py" --prepare "${NEP_PLACEMENT_DOMAIN_FILE}"
        else
            unset NEP_PLACEMENT_METADATA_DIR NEP_PLACEMENT_DOMAIN_FILE
        fi
        graph_override_args=()
        if [[ -n "${CUDA_GRAPH_MODULES_OVERRIDE:-}" ]]; then
            graph_override_args+=(--cuda-graph-modules-override "${CUDA_GRAPH_MODULES_OVERRIDE}")
        fi
        if [[ "${DISABLE_CUDA_GRAPHS:-0}" == 1 ]]; then
            graph_override_args+=(--disable-cuda-graphs)
        fi
        if [[ -n "${EMPTY_UNUSED_MEMORY_LEVEL:-}" ]]; then
            graph_override_args+=(--empty-unused-memory-level "${EMPTY_UNUSED_MEMORY_LEVEL}")
        fi
        if [[ -n "${DDP_NUM_BUCKETS_OVERRIDE:-}" ]]; then
            if [[ ! "${DDP_NUM_BUCKETS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
                echo "DDP_NUM_BUCKETS_OVERRIDE must be a positive integer" >&2
                exit 2
            fi
            ddp_buckets="${DDP_NUM_BUCKETS_OVERRIDE}"
            graph_override_args+=(--ddp-num-buckets-override "${ddp_buckets}")
        fi
        options=$(python3 "${MATRIX}" --repo "${BENCH_REPO}" options "${workload}" "${case_name}" \
            --run-dir "${run_dir}" --train-iters "${TRAIN_ITERS}" \
            --profile-start "${PROFILE_STEP_START}" --profile-end "${PROFILE_STEP_END}" \
            "${graph_override_args[@]}")
        run_nodelist=$(IFS=,; echo "${allocated_nodes[*]:0:run_nodes}")
        master_port=$((32100 + case_index))
        case_index=$((case_index + 1))
        export REPO_DIR="${CODE_REPO}"
        export PRETRAIN_ENTRYPOINT="${ENTRYPOINT}"
        export RANK_LAUNCHER BASE_RANK_LAUNCHER
        export options
        export RUN_WORLD_SIZE="${world_size}"
        export NONUNIFORM_MODE="${mode}"
        export NONUNIFORM_EP_TOPOLOGY="${topology}"
        export TENSOR_MODEL_PARALLEL_SIZE="${tp}"
        export CONTEXT_PARALLEL_SIZE="${cp}"
        export EXPERT_TENSOR_PARALLEL_SIZE="${etp}"
        export SOURCE_HYBRID_EP_DOMAIN_SIZE="${source_hybrid_ep_domain}"
        export MICRO_BATCH_SIZE="${mbs}"
        source_global_batch_size="${gbs}"
        if [[ "${case_name}" == "healthy" ]]; then
            healthy_mbs="${HEALTHY_MICRO_BATCH_SIZE:-${mbs}}"
            healthy_num_microbatches="${HEALTHY_NUM_MICROBATCHES:-${accum}}"
            replica_micro_batch_sizes="${healthy_mbs} ${healthy_mbs}"
            replica_num_microbatches="${healthy_num_microbatches} ${healthy_num_microbatches}"
        else
            full_mbs="${NEP_FULL_MICRO_BATCH_SIZE:-${mbs}}"
            reduced_mbs="${NEP_REDUCED_MICRO_BATCH_SIZE:-${mbs}}"
            full_num_microbatches="${NEP_FULL_NUM_MICROBATCHES:-${accum}}"
            reduced_num_microbatches="${NEP_REDUCED_NUM_MICROBATCHES:-${accum}}"
            replica_micro_batch_sizes="${full_mbs} ${reduced_mbs}"
            replica_num_microbatches="${full_num_microbatches} ${reduced_num_microbatches}"
        fi

        read -r -a topology_units <<< "${topology}"
        read -r -a replica_mbs_values <<< "${replica_micro_batch_sizes}"
        read -r -a replica_num_microbatches_values <<< "${replica_num_microbatches}"
        if ((${#replica_mbs_values[@]} != ${#topology_units[@]} \
            || ${#replica_num_microbatches_values[@]} != ${#topology_units[@]})); then
            echo "${workload}/${case_name}: replica-local batch settings must match topology '${topology}'" >&2
            exit 2
        fi
        gbs=0
        for index in "${!topology_units[@]}"; do
            for value in "${topology_units[index]}" "${replica_mbs_values[index]}" \
                "${replica_num_microbatches_values[index]}"; do
                if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
                    echo "${workload}/${case_name}: batch and topology values must be positive integers" >&2
                    exit 2
                fi
            done
            ((gbs += topology_units[index] * replica_mbs_values[index] * replica_num_microbatches_values[index]))
        done
        export GLOBAL_BATCH_SIZE="${gbs}"
        export TRUE_GLOBAL_BATCH_SIZE="${gbs}"
        export REPLICA_MICRO_BATCH_SIZES="${replica_micro_batch_sizes}"
        export REPLICA_NUM_MICROBATCHES="${replica_num_microbatches}"
        if [[ "${case_name}" == "healthy" ]]; then
            healthy_replica_micro_batch_sizes="${replica_micro_batch_sizes}"
            healthy_replica_num_microbatches="${replica_num_microbatches}"
        else
            nep_replica_micro_batch_sizes="${replica_micro_batch_sizes}"
            nep_replica_num_microbatches="${replica_num_microbatches}"
        fi
        export MASTER_ADDR="${FIRST_NODE}"
        export MASTER_PORT="${master_port}"
        export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${ddp_buckets}"
        if [[ "${workload}" == "nemotron3/ultra" ]]; then
            export NVTE_CPU_OFFLOAD_V1=1
        else
            unset NVTE_CPU_OFFLOAD_V1 || true
        fi
        echo "[training-scripts-nep] starting ${workload}/${case_name}:" \
            "world=${world_size} nodes=${run_nodes} topology='${topology}'" \
            "replica_mbs='${replica_micro_batch_sizes}'" \
            "replica_num_microbatches='${replica_num_microbatches}' gbs=${gbs}" \
            "source_gbs=${source_global_batch_size} buckets=${ddp_buckets}"
        env | sort > "${run_dir}/launch_environment.txt"
        set +e
        # shellcheck disable=SC2016
        timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
            srun --overlap "${cpu_binding_args[@]}" --nodes="${run_nodes}" --nodelist="${run_nodelist}" \
            --ntasks="${world_size}" --ntasks-per-node=4 --kill-on-bad-exit=1 --mpi=none \
            "${container_args[@]}" \
            bash -lc 'export RANK="${SLURM_PROCID}" WORLD_SIZE="${RUN_WORLD_SIZE}" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"; exec bash "${RANK_LAUNCHER}"' \
            2>&1 | tee "${run_dir}/driver.log"
        status=${PIPESTATUS[0]}
        set -e
        if ((status != 0)); then
            failures+=("${workload}/${case_name}:${status}")
            echo "[training-scripts-nep] FAILED ${workload}/${case_name}: status=${status}" >&2
            if [[ "${PROFILE_ACTIVITY_MODE:-}" == nsys || "${PROFILE_ACTIVITY_MODE:-}" == cupti ]]; then
                exit "${status}"
            fi
        else
            echo "[training-scripts-nep] completed ${workload}/${case_name}"
        fi
    done
    if [[ "${CASE_ORDER}" == healthy || "${CASE_ORDER}" == nep ]]; then
        python3 - "${BENCH_REPO}" "${workload}" "${run_dir}" "${TRAIN_ITERS}" "${TIMING_START}" "${TIMING_END}" "${gbs}" "${CASE_ORDER}" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/nonuniform'))
from training_scripts_nep_matrix import analyze_case, workload_by_name
w = workload_by_name(Path(sys.argv[1]), sys.argv[2])
result = analyze_case(w, sys.argv[8], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[7]), int(sys.argv[6]))
(Path(sys.argv[3]) / 'baseline.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
        continue
    fi
    if ! python3 "${MATRIX}" --repo "${BENCH_REPO}" analyze "${workload}" \
        --root "${ROOT_DIR}/${GROUP}" --job-id "${SLURM_JOB_ID}" \
        --train-iters "${TRAIN_ITERS}" --timing-start "${TIMING_START}" \
        --timing-end "${TIMING_END}" \
        --routing-mode "${BENCH_ROUTING_MODE:-random}" \
        --healthy-replica-micro-batch-sizes "${healthy_replica_micro_batch_sizes}" \
        --healthy-replica-num-microbatches "${healthy_replica_num_microbatches}" \
        --nep-replica-micro-batch-sizes "${nep_replica_micro_batch_sizes}" \
        --nep-replica-num-microbatches "${nep_replica_num_microbatches}"; then
        failures+=("${workload}/analysis")
    fi
done

if ((${#failures[@]})); then
    printf '[training-scripts-nep] failures: %s\n' "${failures[*]}" >&2
    exit 1
fi
echo "[training-scripts-nep] all ${GROUP} comparisons passed"
