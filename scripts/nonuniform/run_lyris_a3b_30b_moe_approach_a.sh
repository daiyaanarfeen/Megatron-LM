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
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCHINDUCTOR_WORKER_START="${TORCHINDUCTOR_WORKER_START:-fork}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES="${MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES:-1073741824}"
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW="${MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW:-16}"
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG="${MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG:-0}"

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:25.09}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
RUN_NNODES="${RUN_NNODES:-${SLURM_NNODES}}"
TRAIN_ITERS="${TRAIN_ITERS:-12}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-6}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29760}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS:-16}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-4 4}"
NAME="${NAME:-lyris_a3b_ep8_${NONUNIFORM_EP_TOPOLOGY// /_}_mbs${MICRO_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-8}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
PROFILE="${PROFILE:-1}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"

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
echo "[lyris-a3b] topology=${NONUNIFORM_EP_TOPOLOGY} mbs=${MICRO_BATCH_SIZE} gbs=${GLOBAL_BATCH_SIZE} buckets=${DDP_NUM_BUCKETS}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)

# Compile the dataset helper once into the shared checkout and validate the
# Mamba-specific runtime before starting the distributed workers.
srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" \
    bash -lc "cd '${REPO_DIR}' && python -c 'import mamba_ssm, causal_conv1d, grouped_gemm; from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec; from megatron.core.datasets.utils import compile_helpers; compile_helpers(); print(\"[lyris-a3b] runtime preflight: ok\")'"

profile_args=""
if [[ "${PROFILE}" == "1" ]]; then
    profile_args=" --profile --use-pytorch-profiler --profile-step-start ${PROFILE_STEP_START} --profile-step-end ${PROFILE_STEP_END} --profile-ranks ${PROFILE_RANKS} "
fi

options=" \
    --use-mcore-models \
    --hybrid-layer-pattern MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME \
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
    --num-experts 128 \
    --moe-router-topk 6 \
    --moe-shared-expert-intermediate-size 3712 \
    --moe-token-dispatcher-type alltoall \
    --moe-router-score-function sigmoid \
    --moe-grouped-gemm \
    --moe-aux-loss-coeff 1e-4 \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
    --moe-router-load-balancing-type seq_aux_loss \
    --moe-router-force-load-balancing \
    --moe-permute-fusion \
    --use-fused-weighted-squared-relu \
    --bf16 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
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
    --cuda-graph-impl none \
    --te-rng-tracker \
    --no-load-rng \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 131072 \
    --num-workers 1 \
    --no-create-attention-mask-in-dataloader \
    --overlap-grad-reduce \
    --nonuniform-mode ep \
    --nonuniform-ep-ddp-approach nccl \
    --nonuniform-skip-optimizer-step \
    --nonuniform-ep-num-tp-cp-per-replica ${NONUNIFORM_EP_TOPOLOGY} \
    --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE} \
    --sequence-parallel \
    --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE} \
    --expert-tensor-parallel-size ${EXPERT_TENSOR_PARALLEL_SIZE} \
    --pipeline-model-parallel-size 1 \
    --high-priority-stream-groups ep \
    --ddp-num-buckets ${DDP_NUM_BUCKETS} \
    --attention-backend flash \
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
    ${profile_args} \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 20 \
    --exit-duration-in-mins 40 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    ${EXTRA_MEGATRON_ARGS} "

export RUN_NNODES GPUS_PER_NODE MASTER_ADDR MASTER_PORT REPO_DIR options
echo "[lyris-a3b] launching ${RUN_NNODES} nodes x ${GPUS_PER_NODE} GPUs"
srun --nodes="${RUN_NNODES}" --ntasks="${RUN_NNODES}" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --mpi=none "${container_args[@]}" \
    bash -lc 'cd "${REPO_DIR}" && python -u -m torch.distributed.run \
        --nnodes="${RUN_NNODES}" \
        --nproc-per-node="${GPUS_PER_NODE}" \
        --node-rank="${SLURM_NODEID}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        pretrain_hybrid.py ${options}'
