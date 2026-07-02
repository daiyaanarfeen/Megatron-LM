#!/bin/bash

#SBATCH -p batch
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --dependency=singleton
#SBATCH --job-name=nonuniform_ep_approach_a_smoke

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR="/tmp/triton_cache/"
export MEGATRON_NONUNIFORM_EP_DEBUG="${MEGATRON_NONUNIFORM_EP_DEBUG:-0}"

ASSET_ROOT="${ASSET_ROOT:-/home/scratch.darfeen_gpu}"
ROOT_DIR="${ROOT_DIR:-/home/scratch.darfeen_gpu/training_scripts_dp1_dummy_runs}"
REPO_DIR="${REPO_DIR:-/home/scratch.darfeen_gpu/Megatron-LM-EP}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
NUM_LAYERS="${NUM_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-1024}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
NUM_EXPERTS="${NUM_EXPERTS:-6}"
NAME="${NAME:-nonuniform_ep_approach_a_smoke}"
IMAGE_PATH="${IMAGE_PATH:-${ASSET_ROOT}/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home/scratch.darfeen_gpu:/home/scratch.darfeen_gpu}"
MASTER_PORT="${MASTER_PORT:-29640}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE}}"
TORCH_CUDA_VISIBLE_DEVICES="${TORCH_CUDA_VISIBLE_DEVICES:-}"
NONUNIFORM_EP_DDP_APPROACH="${NONUNIFORM_EP_DDP_APPROACH:-nccl}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-2 2}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-0}"
ENABLE_NSYS_PROFILE="${ENABLE_NSYS_PROFILE:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
PROFILE_RANKS="${PROFILE_RANKS:-0 1 2 3 4 5}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,cublas,cudnn}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-cudaProfilerApi}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"
DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-}"
RUN_DIRECT="${RUN_DIRECT:-0}"
LAUNCHER_MODE="${LAUNCHER_MODE:-torchrun}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"

PROFILE_OPTIONS=""
if [[ "${ENABLE_NSYS_PROFILE}" == "1" ]]; then
    PROFILE_OPTIONS=" \
    --profile \
    --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} \
    --profile-ranks ${PROFILE_RANKS} \
    --nvtx-ranges "
elif [[ "${ENABLE_PYTORCH_PROFILER}" == "1" ]]; then
    PROFILE_OPTIONS=" \
    --profile \
    --use-pytorch-profiler \
    --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} \
    --profile-ranks ${PROFILE_RANKS} "
fi

DDP_BUCKET_OPTIONS=""
if [[ -n "${DDP_BUCKET_SIZE}" ]]; then
    DDP_BUCKET_OPTIONS=" --ddp-bucket-size ${DDP_BUCKET_SIZE} "
fi

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p "${LOGS_DIR}" "${TENSORBOARD_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 2
fi

if [[ ! -f "${IMAGE_PATH}" && "${IMAGE_PATH}" != *"#"* ]]; then
    echo "IMAGE_PATH does not exist: ${IMAGE_PATH}" >&2
    exit 2
fi

options=" \
    --use-mcore-models \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --train-iters ${TRAIN_ITERS} \
    --eval-iters 0 \
    --eval-interval 1000 \
    --log-interval 1 \
    --timing-log-option minmax \
    --attention-backend unfused \
    --no-check-for-nan-in-loss-and-grad \
    --ddp-average-in-collective \
    ${DDP_BUCKET_OPTIONS} \
    --overlap-grad-reduce \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --num-experts ${NUM_EXPERTS} \
    --moe-router-topk 1 \
    --moe-router-pre-softmax \
    --moe-router-force-load-balancing \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE} \
    --moe-grouped-gemm \
    --disable-bias-linear \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 4096 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-decay-style constant \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    ${PROFILE_OPTIONS} \
    --nonuniform-mode ep \
    --nonuniform-ep-num-tp-cp-per-replica ${NONUNIFORM_EP_TOPOLOGY} \
    --nonuniform-ep-ddp-approach ${NONUNIFORM_EP_DDP_APPROACH} \
    ${EXTRA_MEGATRON_ARGS} "

if [[ "${LAUNCHER_MODE}" != "direct" && -z "${TORCH_CUDA_VISIBLE_DEVICES}" && "${NPROC_PER_NODE}" != "${GPUS_PER_NODE}" ]]; then
    TORCH_CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NPROC_PER_NODE - 1))")"
fi

cuda_visible_prefix=""
if [[ -n "${TORCH_CUDA_VISIBLE_DEVICES}" ]]; then
    cuda_visible_prefix="CUDA_VISIBLE_DEVICES=${TORCH_CUDA_VISIBLE_DEVICES} "
fi

nsys_prefix=""
if [[ "${ENABLE_NSYS_PROFILE}" == "1" ]]; then
    NSYS_OUTPUT_DIR="${NSYS_OUTPUT_DIR:-${RUN_DIR}/nsys}"
    NSYS_OUTPUT_NAME="${NSYS_OUTPUT_NAME:-node_\${SLURM_NODEID}}"
    mkdir -p "${NSYS_OUTPUT_DIR}"
    nsys_range_args=""
    if [[ "${NSYS_CAPTURE_RANGE}" != "none" ]]; then
        nsys_range_args="--capture-range=${NSYS_CAPTURE_RANGE} --capture-range-end=stop"
    fi
    nsys_prefix="nsys profile --sample=none --cpuctxsw=none --trace=${NSYS_TRACE} --wait all ${nsys_range_args} --cuda-graph-trace=node --force-overwrite=true --export=sqlite ${NSYS_EXTRA_ARGS} --output=${NSYS_OUTPUT_DIR}/${NSYS_OUTPUT_NAME} "
fi

if [[ "${LAUNCHER_MODE}" == "direct" ]]; then
    run_cmd="cd ${REPO_DIR} && ${cuda_visible_prefix}${nsys_prefix}python -u examples/nonuniform/pretrain_gpt_nonuniform.py ${options}"
else
    run_cmd="cd ${REPO_DIR} && ${cuda_visible_prefix}${nsys_prefix}python -u -m torch.distributed.run --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=\${SLURM_NODEID} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} examples/nonuniform/pretrain_gpt_nonuniform.py ${options}"
fi

echo "[nonuniform_ep_smoke] run_cmd=${run_cmd}"

if [[ "${RUN_DIRECT}" == "1" ]]; then
    sh -c "${run_cmd}"
    exit 0
fi

container_name_options=()
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_name_options=(--container-name "${CONTAINER_NAME}")
fi

srun -l \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    "${container_name_options[@]}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
