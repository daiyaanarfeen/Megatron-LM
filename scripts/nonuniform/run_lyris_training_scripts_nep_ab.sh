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
RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_lyris_training_scripts_nep_rank.sh"
BASE_RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh"
ENTRYPOINT="${CODE_REPO}/examples/nonuniform/pretrain_hybrid_nonuniform.py"
ROOT_DIR="${ROOT_DIR:-${ARTIFACT_ROOT}/slurm_runs/training_scripts_nep_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
TIMING_START="${TIMING_START:-8}"
CASE_TIMEOUT="${CASE_TIMEOUT:-35m}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-}"

mkdir -p "${ROOT_DIR}/${GROUP}" "${ARTIFACT_ROOT}/slurm_runs/lyris"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
FIRST_NODE="${allocated_nodes[0]}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${BENCH_REPO}:${BENCH_REPO},${CODE_REPO}:${CODE_REPO},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}"
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
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
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
    echo "[training-scripts-nep] $(date --iso-8601=seconds) workload=${workload}"
    for case_name in healthy nep; do
        IFS=$'\t' read -r tp cp etp _ mbs accum topology mode world_size run_nodes gbs ddp_buckets segment_nodes source_hybrid_ep_domain < <(
            python3 "${MATRIX}" --repo "${BENCH_REPO}" case-fields "${workload}" "${case_name}"
        )
        if ((run_nodes > ${#allocated_nodes[@]})); then
            echo "${workload}/${case_name}: needs ${run_nodes} nodes, allocation has ${#allocated_nodes[@]}" >&2
            exit 2
        fi
        job_segment=$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*SegmentSize=\([0-9][0-9]*\).*/\1/p')
        if [[ -n "${job_segment}" && "${job_segment}" != "${segment_nodes}" ]]; then
            echo "${workload}/${case_name}: expected segment=${segment_nodes}, got ${job_segment}" >&2
            exit 2
        fi
        run_dir="${ROOT_DIR}/${GROUP}/${slug}/${case_name}/${SLURM_JOB_ID}"
        mkdir -p "${run_dir}/tensorboard" "${run_dir}/torch_profile"
        graph_override_args=()
        if [[ -n "${CUDA_GRAPH_MODULES_OVERRIDE:-}" ]]; then
            graph_override_args+=(--cuda-graph-modules-override "${CUDA_GRAPH_MODULES_OVERRIDE}")
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
        export GLOBAL_BATCH_SIZE="${gbs}"
        export TRUE_GLOBAL_BATCH_SIZE="${gbs}"
        replica_micro_batch_sizes="${mbs} ${mbs}"
        replica_num_microbatches="${accum} ${accum}"
        if [[ "${case_name}" == "nep" && -n "${NEP_REDUCED_MICRO_BATCH_SIZE:-}" ]]; then
            : "${NEP_REDUCED_NUM_MICROBATCHES:?Set with NEP_REDUCED_MICRO_BATCH_SIZE}"
            read -r full_replica_units reduced_replica_units <<< "${topology}"
            override_gbs=$((
                full_replica_units * mbs * accum
                + reduced_replica_units * NEP_REDUCED_MICRO_BATCH_SIZE
                    * NEP_REDUCED_NUM_MICROBATCHES
            ))
            if ((override_gbs != gbs)); then
                echo "${workload}/${case_name}: replica-local batch override produces GBS=${override_gbs}, expected ${gbs}" >&2
                exit 2
            fi
            replica_micro_batch_sizes="${mbs} ${NEP_REDUCED_MICRO_BATCH_SIZE}"
            replica_num_microbatches="${accum} ${NEP_REDUCED_NUM_MICROBATCHES}"
        fi
        export REPLICA_MICRO_BATCH_SIZES="${replica_micro_batch_sizes}"
        export REPLICA_NUM_MICROBATCHES="${replica_num_microbatches}"
        export MASTER_ADDR="${FIRST_NODE}"
        export MASTER_PORT="${master_port}"
        export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${ddp_buckets}"
        if [[ "${workload}" == "nemotron3/ultra" ]]; then
            export NVTE_CPU_OFFLOAD_V1=1
        else
            unset NVTE_CPU_OFFLOAD_V1 || true
        fi
        echo "[training-scripts-nep] starting ${workload}/${case_name}: world=${world_size} nodes=${run_nodes} topology='${topology}' mbs=${mbs} accum=${accum} gbs=${gbs} buckets=${ddp_buckets}"
        set +e
        # shellcheck disable=SC2016
        timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
            srun --overlap --nodes="${run_nodes}" --nodelist="${run_nodelist}" \
            --ntasks="${world_size}" --ntasks-per-node=4 --kill-on-bad-exit=1 --mpi=none \
            "${container_args[@]}" \
            bash -lc 'export RANK="${SLURM_PROCID}" WORLD_SIZE="${RUN_WORLD_SIZE}" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"; exec bash "${RANK_LAUNCHER}"' \
            2>&1 | tee "${run_dir}/driver.log"
        status=${PIPESTATUS[0]}
        set -e
        if ((status != 0)); then
            failures+=("${workload}/${case_name}:${status}")
            echo "[training-scripts-nep] FAILED ${workload}/${case_name}: status=${status}" >&2
        else
            echo "[training-scripts-nep] completed ${workload}/${case_name}"
        fi
    done
    if ! python3 "${MATRIX}" --repo "${BENCH_REPO}" analyze "${workload}" \
        --root "${ROOT_DIR}/${GROUP}" --job-id "${SLURM_JOB_ID}" \
        --train-iters "${TRAIN_ITERS}" --timing-start "${TIMING_START}"; then
        failures+=("${workload}/analysis")
    fi
done

if ((${#failures[@]})); then
    printf '[training-scripts-nep] failures: %s\n' "${failures[*]}" >&2
    exit 1
fi
echo "[training-scripts-nep] all ${GROUP} comparisons passed"
