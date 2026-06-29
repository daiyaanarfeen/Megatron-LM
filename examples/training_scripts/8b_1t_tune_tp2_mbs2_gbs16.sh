#!/bin/bash

#SBATCH -p batch
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH --mem=0
# GB200/GB300 have 4 GPUs/node; set --ntasks-per-node=4, --gpus-per-node=4, and
# add --segment=4 (or --segment=16 for 16-node segments) on those platforms.
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --dependency=singleton
#SBATCH --job-name=8b_1t_tune_tp2_mbs2_gbs16

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.

# Short DP1 dummy-data benchmark defaults for this cluster.
ASSET_ROOT="${ASSET_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/dnarayanan/bf16rs_technical_report}"
ROOT_DIR="${ROOT_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/training_scripts_dp1_dummy_runs}"
REPO_DIR="${REPO_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_sysarch/users/darfeen/Megatron-LM-EP}"
TRAIN_ITERS="${TRAIN_ITERS:-50}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-10}"
# Run name; change this per experiment.
NAME="8b_1t_tune_tp2_mbs2_gbs16"
IMAGE_PATH="${IMAGE_PATH:-${ASSET_ROOT}/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

mkdir -p ${LOGS_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}

options=" \
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 21504 \
    --kv-channels 128 \
    --squared-relu \
    --untie-embeddings-and-output-weights \
    --init-method-std 0.014 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --rotary-percent 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --disable-bias-linear \
    --normalization RMSNorm \
    \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style WSD \
    --lr-decay-iters ${TRAIN_ITERS} \
    --lr-warmup-iters 1 \
    --lr-wsd-decay-style minus_sqrt \
    --lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS} \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-interval 1000 \
    --eval-iters 0 \
    \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --ddp-num-buckets 8 \
    --attention-backend flash \
    \
    --log-interval 1 \
    --log-memory-interval 50 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --log-progress \
    --log-energy \
    --logging-level 20 \
    --timing-log-option minmax \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 10 \
    --exit-duration-in-mins 55 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 "

run_cmd="python -u ${REPO_DIR}/pretrain_gpt.py ${options}"

# Adjust --container-mounts below if ROOT_DIR lives on a different filesystem
# (e.g. "/scratch:/scratch" on clusters where assets live under /scratch).
srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "/lustre:/lustre" \
    --no-container-mount-home \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
