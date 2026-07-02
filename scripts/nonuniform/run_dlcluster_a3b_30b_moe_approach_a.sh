#!/bin/bash

# dlcluster wrapper for examples/training_scripts/a3b_30b_moe_1t_approach_a_nccl.sh.
#
# The live checkout is not visible from dlcluster compute nodes, so this stages a
# repo archive and the PyTorch sqsh image to node-local /tmp, runs the benchmark
# from the staged repo, and copies logs back to the login filesystem.

#SBATCH --account=blackwell
#SBATCH --partition=gb200nvl72
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/%x-%j.out
#SBATCH --error=/tmp/%x-%j.err
#SBATCH --job-name=dl_a3b_moe_nep

set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export TORCHINDUCTOR_WORKER_START="${TORCHINDUCTOR_WORKER_START:-fork}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES="${MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES:-1073741824}"
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW="${MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW:-16}"
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG="${MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG:-0}"

LOGIN_HOST="${LOGIN_HOST:-dlcluster-login-01}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/darfeen_known_hosts}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/darfeen_mep_a3b}"
REPO_ARCHIVE_SRC="${REPO_ARCHIVE_SRC:-${LOGIN_HOST}:/tmp/megatron_ep_stage.tar.gz}"
IMAGE_SRC="${IMAGE_SRC:-${LOGIN_HOST}:/home/scratch.darfeen_gpu/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
IMAGE_BYTES="${IMAGE_BYTES:-29622525952}"
RESULTS_BASE_PATH="${RESULTS_BASE_PATH:-/home/scratch.darfeen_gpu/dlcluster_runs}"

CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"
IMAGE_PATH="${STAGE_ROOT}/image.sqsh"
REPO_DIR="${STAGE_ROOT}/repo"
ROOT_DIR="${STAGE_ROOT}/runs"
RUN_NNODES="${RUN_NNODES:-${SLURM_NNODES}}"
RUN_GPUS_PER_NODE="${RUN_GPUS_PER_NODE:-4}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-10}"
MASTER_PORT="${MASTER_PORT:-29710}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-16 16}"
NAME="${NAME:-dl_a3b_30b_moe_approach_a_${NONUNIFORM_EP_TOPOLOGY// /_}_t${TRAIN_ITERS}}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
EXPERT_MODEL_PARALLEL_SIZE="${EXPERT_MODEL_PARALLEL_SIZE:-32}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2688}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-32}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
MAMBA_NUM_HEADS="${MAMBA_NUM_HEADS:-64}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-1856}"
KV_CHANNELS="${KV_CHANNELS:-128}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-6}"
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE="${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-3712}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
VOCAB_SIZE="${VOCAB_SIZE:-131072}"
CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL="${CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL:-}"
PROFILE="${PROFILE:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"

hash_check_args=""
if [[ -n "${CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL}" ]]; then
    hash_check_args=" --check-weight-hash-across-dp-replicas-interval ${CHECK_WEIGHT_HASH_ACROSS_DP_REPLICAS_INTERVAL} "
fi

profile_args=""
if [[ "${PROFILE}" == "1" ]]; then
    profile_args=" --profile --use-pytorch-profiler --profile-step-start ${PROFILE_STEP_START} --profile-step-end ${PROFILE_STEP_END} --profile-ranks ${PROFILE_RANKS} "
fi

RUN_DIR="${ROOT_DIR}/${NAME}"
LOGS_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${ROOT_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
TORCH_PROFILE_DIR="${RUN_DIR}/torch_profile"
DRIVER_LOG="${STAGE_ROOT}/driver_${SLURM_JOB_ID}.log"

mkdir -p "${STAGE_ROOT}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

copy_logs_back() {
    status=$?
    set +e
    echo "[dlcluster-a3b] exit status: ${status}"
    echo "[dlcluster-a3b] copying logs back to ${LOGIN_HOST}:${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}/"
    tar -czf "${STAGE_ROOT}/logs_${SLURM_JOB_ID}.tgz" -C "${STAGE_ROOT}" \
        "$(basename "${DRIVER_LOG}")" \
        "runs/${NAME}/logs" \
        "runs/${NAME}/tensorboard" \
        "runs/${NAME}/torch_profile" 2>/dev/null
    ssh ${SSH_OPTS} "${LOGIN_HOST}" "mkdir -p '${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}'"
    rsync -av -e "ssh ${SSH_OPTS}" "${STAGE_ROOT}/logs_${SLURM_JOB_ID}.tgz" \
        "${LOGIN_HOST}:${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}/"
    exit "${status}"
}
trap copy_logs_back EXIT

echo "[dlcluster-a3b] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[dlcluster-a3b] stage_root=${STAGE_ROOT}"
echo "[dlcluster-a3b] topology=${NONUNIFORM_EP_TOPOLOGY} gbs=${GLOBAL_BATCH_SIZE}"

stage_one_node='
set -euo pipefail
mkdir -p "'"${STAGE_ROOT}"'" "'"${ROOT_DIR}"'" "'"${LOGS_DIR}"'" "'"${CHECKPOINT_DIR}"'" "'"${DATACACHE_DIR}"'" "'"${TENSORBOARD_DIR}"'"
retry_cmd() {
    local status=0
    local attempt=1
    while [[ "${attempt}" -le 5 ]]; do
        if "$@"; then
            return 0
        fi
        status=$?
        echo "[stage $(hostname)] attempt ${attempt}/5 failed with status ${status}: $*"
        sleep $((attempt * 10))
        attempt=$((attempt + 1))
    done
    return "${status}"
}
export ENROOT_CACHE_PATH="'"${STAGE_ROOT}"'/enroot-cache"
export ENROOT_DATA_PATH="'"${STAGE_ROOT}"'/enroot-data"
export ENROOT_RUNTIME_PATH="'"${STAGE_ROOT}"'/enroot-runtime"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"
container_exists=0
if enroot list | grep -qx "'"${CONTAINER_NAME}"'"; then
    container_exists=1
fi
if [[ "${container_exists}" != "1" ]] && { [[ ! -f "'"${IMAGE_PATH}"'" ]] || [[ "$(stat -c%s "'"${IMAGE_PATH}"'" 2>/dev/null || echo 0)" != "'"${IMAGE_BYTES}"'" ]]; }; then
    echo "[stage $(hostname)] syncing image"
    retry_cmd rsync -av --partial --append-verify --info=progress2 -e "ssh '"${SSH_OPTS}"'" "'"${IMAGE_SRC}"'" "'"${IMAGE_PATH}"'"
elif [[ "${container_exists}" == "1" ]]; then
    echo "[stage $(hostname)] enroot container exists; skipping image sync"
fi
echo "[stage $(hostname)] syncing repo archive"
retry_cmd rsync -av -e "ssh '"${SSH_OPTS}"'" "'"${REPO_ARCHIVE_SRC}"'" "'"${STAGE_ROOT}"'/repo.tar.gz"
rm -rf "'"${REPO_DIR}"'"
mkdir -p "'"${REPO_DIR}"'"
tar -xzf "'"${STAGE_ROOT}"'/repo.tar.gz" -C "'"${REPO_DIR}"'" --strip-components=1
if ! enroot list | grep -qx "'"${CONTAINER_NAME}"'"; then
    echo "[stage $(hostname)] creating enroot container "'"${CONTAINER_NAME}"'""
    enroot create --name "'"${CONTAINER_NAME}"'" "'"${IMAGE_PATH}"'"
else
    echo "[stage $(hostname)] enroot container already exists"
fi
if ! compgen -G "'"${REPO_DIR}"'/megatron/core/datasets/helpers_cpp*.so" >/dev/null; then
    echo "[stage $(hostname)] compiling dataset helper extension"
    enroot start --rw --mount "'"${STAGE_ROOT}"':'"${STAGE_ROOT}"'" "'"${CONTAINER_NAME}"'" \
        sh -lc "cd '"${REPO_DIR}"' && python -c \"from megatron.core.datasets.utils import compile_helpers; compile_helpers()\""
else
    echo "[stage $(hostname)] dataset helper extension already exists"
fi
'

echo "[dlcluster-a3b] staging all nodes"
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --mpi=none \
    bash -lc "${stage_one_node}"

mkdir -p "${LOGS_DIR}" "${CHECKPOINT_DIR}" "${DATACACHE_DIR}" "${TENSORBOARD_DIR}" "${TORCH_PROFILE_DIR}"

options=" \
    --use-mcore-models \
    --hybrid-layer-pattern ${HYBRID_LAYER_PATTERN} \
    --spec megatron.core.models.hybrid.hybrid_layer_specs hybrid_stack_spec \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --group-query-attention \
    --num-query-groups ${NUM_QUERY_GROUPS} \
    --mamba-num-heads ${MAMBA_NUM_HEADS} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --kv-channels ${KV_CHANNELS} \
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
    --moe-shared-expert-intermediate-size ${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE} \
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
    --cuda-graph-impl none \
    --te-rng-tracker \
    --no-load-rng \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size ${VOCAB_SIZE} \
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
    --ddp-num-buckets 8 \
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
    ${hash_check_args} \
    ${profile_args} \
    --manual-gc \
    --manual-gc-interval 10 \
    --distributed-timeout-minutes 20 \
    --exit-duration-in-mins 55 \
    --disable-gloo-process-groups \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    ${EXTRA_MEGATRON_ARGS} "

run_cmd="cd ${REPO_DIR} && python -u ${REPO_DIR}/pretrain_hybrid.py ${options}"

echo "[dlcluster-a3b] launching benchmark: nodes=${RUN_NNODES} gpus_per_node=${RUN_GPUS_PER_NODE} train_iters=${TRAIN_ITERS}"
export ENROOT_CACHE_PATH="${STAGE_ROOT}/enroot-cache"
export ENROOT_DATA_PATH="${STAGE_ROOT}/enroot-data"
export ENROOT_RUNTIME_PATH="${STAGE_ROOT}/enroot-runtime"
srun -l \
    --nodes="${RUN_NNODES}" \
    --ntasks="$((RUN_NNODES * RUN_GPUS_PER_NODE))" \
    --ntasks-per-node="${RUN_GPUS_PER_NODE}" \
    --gpus-per-node="${RUN_GPUS_PER_NODE}" \
    --mpi=none \
    --container-image "${IMAGE_PATH}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${STAGE_ROOT}:${STAGE_ROOT}" \
    --no-container-mount-home \
    sh -c "${run_cmd}"
