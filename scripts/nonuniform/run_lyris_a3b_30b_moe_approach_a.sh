#!/bin/bash

# Lyris GB200 wrapper for the profiled a3b_30b_moe_1t Approach-A sweep.
# The checkout and results live on shared /home, so no per-node repo/image
# staging or login-node copy-back is needed.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=lyris_a3b_moe_nep

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_LAUNCH_ORDER_IMPLICIT="${NCCL_LAUNCH_ORDER_IMPLICIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER="${MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER:-0}"
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK="${MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK:-0}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCHINDUCTOR_WORKER_START="${TORCHINDUCTOR_WORKER_START:-fork}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES="${MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES:-1073741824}"
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS="${MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS:-}"
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS="${MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS:-1}"
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW="${MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW:-16}"
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS:-3}"
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG="${MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG:-0}"
export MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD="${MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD:-0}"
export MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER="${MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER:-0}"

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:25.09}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RUN_NNODES="${RUN_NNODES:-${SLURM_NNODES}}"
RUN_WORLD_SIZE="${RUN_WORLD_SIZE:-$((RUN_NNODES * GPUS_PER_NODE))}"
USE_DIRECT_SRUN_RANKS="${USE_DIRECT_SRUN_RANKS:-0}"
DIRECT_RANK_LAUNCHER="${DIRECT_RANK_LAUNCHER:-${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh}"
TRAIN_ITERS="${TRAIN_ITERS:-12}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-6}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29760}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
REPLICA_MICRO_BATCH_SIZES="${REPLICA_MICRO_BATCH_SIZES:-}"
REPLICA_NUM_MICROBATCHES="${REPLICA_NUM_MICROBATCHES:-}"
TRUE_GLOBAL_BATCH_SIZE="${TRUE_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS:-16}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-4 4}"
NAME="${NAME:-lyris_a3b_ep8_${NONUNIFORM_EP_TOPOLOGY// /_}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
PROFILE="${PROFILE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
HIGH_PRIORITY_STREAM_GROUPS="${HIGH_PRIORITY_STREAM_GROUPS-ep}"
CUDA_GRAPH_IMPL="${CUDA_GRAPH_IMPL:-none}"
NONUNIFORM_MODE="${NONUNIFORM_MODE:-ep}"
NONUNIFORM_SKIP_OPTIMIZER_STEP="${NONUNIFORM_SKIP_OPTIMIZER_STEP:-1}"
USE_GLOO_PROCESS_GROUPS="${USE_GLOO_PROCESS_GROUPS:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-20}"
EXIT_DURATION_IN_MINS="${EXIT_DURATION_IN_MINS:-40}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME}"
MOE_ROUTER_FORCE_LOAD_BALANCING="${MOE_ROUTER_FORCE_LOAD_BALANCING:-1}"
MOE_ROUTER_FORCE_UNIFORM_ROUTING="${MOE_ROUTER_FORCE_UNIFORM_ROUTING:-0}"
MOE_ROUTER_FORCE_BIASED="${MOE_ROUTER_FORCE_BIASED:-}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-6}"
LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-1}"
LOG_NUM_ZEROS_IN_GRAD="${LOG_NUM_ZEROS_IN_GRAD:-1}"
LOG_ENERGY="${LOG_ENERGY:-0}"
MANUAL_GC_INTERVAL="${MANUAL_GC_INTERVAL:-10}"

for toggle in MOE_ROUTER_FORCE_LOAD_BALANCING MOE_ROUTER_FORCE_UNIFORM_ROUTING NONUNIFORM_SKIP_OPTIMIZER_STEP; do
    case "${!toggle}" in
        0|1) ;;
        *)
            echo "${toggle} must be 0 or 1" >&2
            exit 2
            ;;
    esac
done
if [[ "${MOE_ROUTER_FORCE_LOAD_BALANCING}" == "1" && -n "${MOE_ROUTER_FORCE_BIASED}" ]] || \
    [[ "${MOE_ROUTER_FORCE_UNIFORM_ROUTING}" == "1" && \
       ("${MOE_ROUTER_FORCE_LOAD_BALANCING}" == "1" || -n "${MOE_ROUTER_FORCE_BIASED}") ]]; then
    echo "Forced random, exact-uniform, and biased routing modes are mutually exclusive" >&2
    exit 2
fi
if ! [[ "${MOE_ROUTER_TOPK}" =~ ^[1-9][0-9]*$ ]] || ((MOE_ROUTER_TOPK > NUM_EXPERTS)); then
    echo "MOE_ROUTER_TOPK must be an integer in [1, NUM_EXPERTS]" >&2
    exit 2
fi
for toggle in LOG_PARAMS_NORM LOG_NUM_ZEROS_IN_GRAD LOG_ENERGY; do
    case "${!toggle}" in
        0|1) ;;
        *)
            echo "${toggle} must be 0 or 1" >&2
            exit 2
            ;;
    esac
done
if ! [[ "${MANUAL_GC_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MANUAL_GC_INTERVAL must be a positive integer" >&2
    exit 2
fi

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((RUN_NNODES > ${#allocated_nodes[@]})); then
    echo "RUN_NNODES=${RUN_NNODES} exceeds the ${#allocated_nodes[@]} allocated nodes" >&2
    exit 2
fi
if ((RUN_WORLD_SIZE > RUN_NNODES * GPUS_PER_NODE)); then
    echo "RUN_WORLD_SIZE=${RUN_WORLD_SIZE} exceeds the ${RUN_NNODES} x ${GPUS_PER_NODE} GPU launch capacity" >&2
    exit 2
fi
run_nodes=("${allocated_nodes[@]:0:RUN_NNODES}")
RUN_NODELIST=$(IFS=,; echo "${run_nodes[*]}")

RUN_DIR="${ROOT_DIR}/${NAME}/${SLURM_JOB_ID}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
TORCH_PROFILE_DIR="${RUN_DIR}/torch_profile"
DRIVER_LOG="${RUN_DIR}/driver_${SLURM_JOB_ID}.log"

mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" \
    "${TENSORBOARD_DIR}" "${TORCH_PROFILE_DIR}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

finish() {
    status=$?
    echo "[lyris-a3b] exit status: ${status}"
    exit "${status}"
}
trap finish EXIT

echo "[lyris-a3b] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} image=${IMAGE}"
echo "[lyris-a3b] topology=${NONUNIFORM_EP_TOPOLOGY} experts=${NUM_EXPERTS} mbs=${MICRO_BATCH_SIZE} gbs=${GLOBAL_BATCH_SIZE} seq=${SEQ_LENGTH} buckets=${DDP_NUM_BUCKETS} pattern=${HYBRID_LAYER_PATTERN}"
echo "[lyris-a3b] run_nodes=${RUN_NODELIST}"
echo "[lyris-a3b] direct=${USE_DIRECT_SRUN_RANKS} world=${RUN_WORLD_SIZE} true_gbs=${TRUE_GLOBAL_BATCH_SIZE} replica_mbs='${REPLICA_MICRO_BATCH_SIZES}' replica_num_microbatches='${REPLICA_NUM_MICROBATCHES}'"
echo "[lyris-a3b] zero_sm=${MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD} skip_scatter=${MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER} skip_owner_grad_check=${MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK} cuda_connections=${CUDA_DEVICE_MAX_CONNECTIONS} nccl_implicit_order=${NCCL_LAUNCH_ORDER_IMPLICIT} torch_nccl_blocking_wait=${TORCH_NCCL_BLOCKING_WAIT}"
echo "[lyris-a3b] target_chunks=${MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS:-auto} scatter_chunks=${MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS} chunk_window=${MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW} max_gather_bytes=${MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES} expert_bucket_groups=${MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS} a2a_scatter_scheduler=${MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER}"
echo "[lyris-a3b] router_topk=${MOE_ROUTER_TOPK} router_force_balance=${MOE_ROUTER_FORCE_LOAD_BALANCING} router_force_uniform=${MOE_ROUTER_FORCE_UNIFORM_ROUTING} router_force_biased=${MOE_ROUTER_FORCE_BIASED:-none}"
echo "[lyris-a3b] log_params_norm=${LOG_PARAMS_NORM} log_num_zeros_in_grad=${LOG_NUM_ZEROS_IN_GRAD} log_energy=${LOG_ENERGY} manual_gc_interval=${MANUAL_GC_INTERVAL}"
echo "[lyris-a3b] nonuniform_skip_optimizer_step=${NONUNIFORM_SKIP_OPTIMIZER_STEP}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

# Compile the dataset helper once into the shared checkout and validate the
# Mamba-specific runtime before starting the distributed workers.
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
    srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" \
        bash -lc "cd '${REPO_DIR}' && python -c 'import mamba_ssm, causal_conv1d; from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec; from megatron.core.datasets.utils import compile_helpers; compile_helpers(); print(\"[lyris-a3b] runtime preflight: ok\")'"
fi

profile_args=""
if [[ "${PROFILE}" == "1" ]]; then
    profile_args=" --profile --use-pytorch-profiler --profile-step-start ${PROFILE_STEP_START} --profile-step-end ${PROFILE_STEP_END} --profile-ranks ${PROFILE_RANKS} "
fi

high_priority_stream_args=""
if [[ -n "${HIGH_PRIORITY_STREAM_GROUPS}" ]]; then
    high_priority_stream_args=" --high-priority-stream-groups ${HIGH_PRIORITY_STREAM_GROUPS} "
fi

nonuniform_args=" --nonuniform-mode ${NONUNIFORM_MODE} "
if [[ "${NONUNIFORM_SKIP_OPTIMIZER_STEP}" == "1" ]]; then
    nonuniform_args+=" --nonuniform-skip-optimizer-step "
fi
if [[ "${NONUNIFORM_MODE}" == "ep" ]]; then
    nonuniform_args+=" --nonuniform-ep-ddp-approach nccl --nonuniform-ep-num-tp-cp-per-replica ${NONUNIFORM_EP_TOPOLOGY} "
elif [[ "${NONUNIFORM_MODE}" != "none" ]]; then
    echo "Unsupported NONUNIFORM_MODE=${NONUNIFORM_MODE}" >&2
    exit 2
fi

gloo_args=""
if [[ "${USE_GLOO_PROCESS_GROUPS}" != "1" ]]; then
    gloo_args=" --disable-gloo-process-groups "
fi

case "${CUDA_GRAPH_IMPL}" in
    none)
        cuda_graph_args=" --cuda-graph-impl none "
        ;;
    local)
        cuda_graph_args=" --cuda-graph-impl local --cuda-graph-modules mamba attn moe_router "
        ;;
    *)
        echo "Unsupported CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL}" >&2
        exit 2
        ;;
esac

moe_router_benchmark_args=""
if [[ "${MOE_ROUTER_FORCE_LOAD_BALANCING}" == "1" ]]; then
    moe_router_benchmark_args=" --moe-router-force-load-balancing "
fi
if [[ "${MOE_ROUTER_FORCE_UNIFORM_ROUTING}" == "1" ]]; then
    moe_router_benchmark_args+=" --moe-router-force-uniform-routing "
fi
if [[ -n "${MOE_ROUTER_FORCE_BIASED}" ]]; then
    moe_router_benchmark_args+=" --moe-router-force-biased=${MOE_ROUTER_FORCE_BIASED} "
fi

diagnostic_logging_args=""
if [[ "${LOG_PARAMS_NORM}" == "1" ]]; then
    diagnostic_logging_args+=" --log-params-norm "
fi
if [[ "${LOG_NUM_ZEROS_IN_GRAD}" == "1" ]]; then
    diagnostic_logging_args+=" --log-num-zeros-in-grad "
fi
if [[ "${LOG_ENERGY}" == "1" ]]; then
    diagnostic_logging_args+=" --log-energy "
fi

options=" \
    --use-mcore-models \
    --hybrid-layer-pattern ${HYBRID_LAYER_PATTERN} \
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
    --hidden-size 2688 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 2 \
    --mamba-num-heads 64 \
    --ffn-hidden-size 1856 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.0173 \
    --position-embedding-type none \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk ${MOE_ROUTER_TOPK} \
    --moe-shared-expert-intermediate-size 3712 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    ${moe_router_benchmark_args} \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --bf16 \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style WSD \
    --lr-decay-iters ${TRAIN_ITERS} \
    --lr-warmup-iters 1 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --lr 1.2e-3 \
    --min-lr 1.2e-5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 0 \
    ${cuda_graph_args} \
    --te-rng-tracker \
    --no-load-rng \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    --overlap-grad-reduce \
    ${nonuniform_args} \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    --sequence-parallel \
    --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE} \
    --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE} \
    --pipeline-model-parallel-size 1 \
    ${high_priority_stream_args} \
    --ddp-num-buckets ${DDP_NUM_BUCKETS} \
    --attention-backend flash \
    --log-interval 1 \
    --log-memory-interval 50 \
    ${diagnostic_logging_args} \
    --log-throughput \
    --log-progress \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${profile_args} \
    --manual-gc \
    --manual-gc-interval ${MANUAL_GC_INTERVAL} \
    --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES} \
    --exit-duration-in-mins ${EXIT_DURATION_IN_MINS} \
    ${gloo_args} \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    ${EXTRA_MEGATRON_ARGS} "

export RUN_NNODES RUN_WORLD_SIZE GPUS_PER_NODE MASTER_ADDR MASTER_PORT REPO_DIR options
export NONUNIFORM_EP_TOPOLOGY TENSOR_MODEL_PARALLEL_SIZE CONTEXT_PARALLEL_SIZE
export MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TRUE_GLOBAL_BATCH_SIZE
export REPLICA_MICRO_BATCH_SIZES REPLICA_NUM_MICROBATCHES DIRECT_RANK_LAUNCHER
echo "[lyris-a3b] launching ${RUN_WORLD_SIZE} ranks on ${RUN_NNODES} nodes x ${GPUS_PER_NODE} GPUs"
if [[ "${USE_DIRECT_SRUN_RANKS}" == "1" ]]; then
    srun --overlap --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
        --ntasks="${RUN_WORLD_SIZE}" --ntasks-per-node="${GPUS_PER_NODE}" \
        --kill-on-bad-exit=1 --mpi=none "${container_args[@]}" \
        bash -lc 'export RANK="${SLURM_PROCID}" WORLD_SIZE="${RUN_WORLD_SIZE}" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"; exec bash "${DIRECT_RANK_LAUNCHER}"'
else
    if ((RUN_WORLD_SIZE != RUN_NNODES * GPUS_PER_NODE)); then
        echo "Rectangular torchrun launch requires RUN_WORLD_SIZE=RUN_NNODES*GPUS_PER_NODE" >&2
        exit 2
    fi
    srun --overlap --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
        --ntasks="${RUN_NNODES}" --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --mpi=none "${container_args[@]}" \
        bash -lc 'cd "${REPO_DIR}" && python -u -m torch.distributed.run \
            --nnodes="${RUN_NNODES}" \
            --nproc-per-node="${GPUS_PER_NODE}" \
            --node-rank="${SLURM_NODEID}" \
            --master-addr="${MASTER_ADDR}" \
            --master-port="${MASTER_PORT}" \
            pretrain_hybrid.py ${options}'
fi
