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

ASSET_ROOT="${ASSET_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/dnarayanan/bf16rs_technical_report}"
ROOT_DIR="${ROOT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs}"
REPO_DIR="${REPO_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/Megatron-LM-EP}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
PROFILE_STEP_START="${PROFILE_STEP_START:-0}"
PROFILE_STEP_END="${PROFILE_STEP_END:-2}"
NAME="${NAME:-nonuniform_ep_approach_a_smoke_profile}"
IMAGE_PATH="${IMAGE_PATH:-${ASSET_ROOT}/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"
MASTER_PORT="${MASTER_PORT:-29641}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NONUNIFORM_EP_DDP_APPROACH="${NONUNIFORM_EP_DDP_APPROACH:-nccl}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-3 1}"
MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
mkdir -p "${LOGS_DIR}" "${TENSORBOARD_DIR}"

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
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
