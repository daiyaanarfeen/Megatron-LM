#!/bin/bash

#SBATCH -p batch
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 00:15:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --dependency=singleton
#SBATCH --job-name=nonuniform_ep_a_smoke_prof

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR="/tmp/triton_cache/"

ASSET_ROOT="${ASSET_ROOT:-/home/scratch.darfeen_gpu}"
ROOT_DIR="${ROOT_DIR:-/home/scratch.darfeen_gpu/training_scripts_dp1_dummy_runs}"
REPO_DIR="${REPO_DIR:-/home/scratch.darfeen_gpu/Megatron-LM-EP}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
PROFILE_STEP_START="${PROFILE_STEP_START:-0}"
PROFILE_STEP_END="${PROFILE_STEP_END:-2}"
NAME="${NAME:-nonuniform_ep_approach_a_smoke_profile}"
IMAGE_PATH="${IMAGE_PATH:-${ASSET_ROOT}/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/home/scratch.darfeen_gpu:/home/scratch.darfeen_gpu}"
MASTER_PORT="${MASTER_PORT:-29641}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NONUNIFORM_EP_DDP_APPROACH="${NONUNIFORM_EP_DDP_APPROACH:-nccl}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-2 2}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p "${LOGS_DIR}" "${TENSORBOARD_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "REPO_DIR does not exist: ${REPO_DIR}" >&2
    exit 2
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "IMAGE_PATH does not exist: ${IMAGE_PATH}" >&2
    exit 2
fi

options=" \
    --use-mcore-models \
    --num-layers 2 \
    --hidden-size 256 \
    --ffn-hidden-size 1024 \
    --num-attention-heads 4 \
    --seq-length 128 \
    --max-position-embeddings 128 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --train-iters ${TRAIN_ITERS} \
    --eval-iters 0 \
    --eval-interval 1000 \
    --log-interval 1 \
    --timing-log-option minmax \
    --attention-backend unfused \
    --no-check-for-nan-in-loss-and-grad \
    --ddp-average-in-collective \
    --overlap-grad-reduce \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --num-experts 6 \
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
    --profile \
    --use-pytorch-profiler \
    --profile-step-start ${PROFILE_STEP_START} \
    --profile-step-end ${PROFILE_STEP_END} \
    --profile-ranks 0 1 2 3 \
    --nonuniform-mode ep \
    --nonuniform-ep-num-tp-cp-per-replica ${NONUNIFORM_EP_TOPOLOGY} \
    --nonuniform-ep-ddp-approach ${NONUNIFORM_EP_DDP_APPROACH} "

run_cmd="cd ${REPO_DIR} && python -u -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=${MASTER_PORT} examples/nonuniform/pretrain_gpt_nonuniform.py ${options}"

srun -l \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
