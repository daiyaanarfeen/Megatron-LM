#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=16
#SBATCH --time=01:00:00
#SBATCH --job-name=nep_ep32_28
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Standard GPT comparison for the shared NTP/NEP opt-in branch:
# - uniform baseline: two uniform EP32 replicas, TP2, CP2, ETP1 on 64 ranks
# - nonuniform NEP: EP32/EP28 replicas, TP2, CP2, ETP1 on 60 ranks
# - proportional samples: uniform GBS=32, nonuniform GBS=30 by default
# - profiled defaults use longer sequence length, wider experts, recompute, and large
#   DDP buckets to keep the run compute-heavy with <=5 expert grad buckets.

set -euo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
LOGDIR=${LOGDIR:-$WORKDIR/nonuniform_ep32_28_logs}
MASTER_PORT=${MASTER_PORT:-29620}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAIN_ITERS=${TRAIN_ITERS:-10}
RUN_NONUNIFORM=${RUN_NONUNIFORM:-1}
RUN_UNIFORM=${RUN_UNIFORM:-1}

UNIFORM_GBS=${UNIFORM_GBS:-32}
NONUNIFORM_GBS=${NONUNIFORM_GBS:-30}
SEQ_LENGTH=${SEQ_LENGTH:-16384}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
NUM_LAYERS=${NUM_LAYERS:-2}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-24576}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_EXPERTS=${NUM_EXPERTS:-224}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
DDP_BUCKET_SIZE=${DDP_BUCKET_SIZE:-700000000}
PROFILE=${PROFILE:-0}
PROFILE_STEP_START=${PROFILE_STEP_START:-5}
PROFILE_STEP_END=${PROFILE_STEP_END:-7}
PROFILE_RANKS=${PROFILE_RANKS:-0}

HSG_CACHED_IMAGE=${HSG_CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.08-nvshmem-megatron-het-ep.sqsh}

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}

if [[ -z "${IMAGE:-}" && -f "$HSG_CACHED_IMAGE" ]]; then
  IMAGE=$HSG_CACHED_IMAGE
fi

mkdir -p "$LOGDIR"
status=0

common_args=(
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 4096
  --train-iters "$TRAIN_ITERS"
  --eval-iters 0
  --eval-interval "$TRAIN_ITERS"
  --log-interval 1
  --timing-log-level 1
  --timing-log-option minmax
  --no-check-for-nan-in-loss-and-grad
  --ddp-average-in-collective
  --overlap-grad-reduce
  --bf16
  --grad-reduce-in-bf16
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers "$RECOMPUTE_NUM_LAYERS"
  --tensor-model-parallel-size 2
  --context-parallel-size 2
  --expert-tensor-parallel-size 1
  --sequence-parallel
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --num-attention-heads "$NUM_ATTENTION_HEADS"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --num-experts "$NUM_EXPERTS"
  --moe-router-topk 2
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 0.01
  --moe-token-dispatcher-type alltoall
  --moe-grouped-gemm
  --disable-bias-linear
  --transformer-impl transformer_engine
  --lr 1.0e-4
  --min-lr 1.0e-5
  --lr-decay-style constant
)

if [[ -n "$DDP_BUCKET_SIZE" ]]; then
  common_args+=(
    --ddp-bucket-size "$DDP_BUCKET_SIZE"
  )
fi

make_profile_args() {
  local name=$1
  PROFILE_ARGS=()
  if [[ "$PROFILE" == "1" ]]; then
    local profile_ranks=()
    read -r -a profile_ranks <<< "$PROFILE_RANKS"
    PROFILE_ARGS=(
      --profile
      --use-pytorch-profiler
      --profile-step-start "$PROFILE_STEP_START"
      --profile-step-end "$PROFILE_STEP_END"
      --profile-ranks "${profile_ranks[@]}"
      --tensorboard-dir "$LOGDIR/${name}_tensorboard"
    )
  fi
}

run_in_allocation() {
  local nnodes=$1
  local log_file=$2
  local master_port=$3
  shift 3
  local torchrun_args=("$@")
  local master
  master=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
  master=$(getent hosts "$master" | awk '{print $1}' | head -n1)
  master=${master:-$(scontrol show hostname "$SLURM_NODELIST" | head -n1)}

  local srun_cmd=(srun --nodes="$nnodes" --ntasks="$nnodes" --ntasks-per-node=1)
  if [[ -n "${IMAGE:-}" ]]; then
    srun_cmd+=(
      --container-image="$IMAGE"
      --container-mounts="$WORKDIR:$WORKDIR"
      --container-workdir="$WORKDIR"
      --container-env=NCCL_NVLS_ENABLE,NCCL_LAUNCH_ORDER_IMPLICIT
      --container-env=CUDA_DEVICE_MAX_CONNECTIONS,UCX_NET_DEVICES
    )
  fi

  "${srun_cmd[@]}" bash -lc "
    set -euo pipefail
    ulimit -s 8192
    cd '$WORKDIR'
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$nnodes \
      --node_rank=\$SLURM_NODEID --master_addr=$master --master_port=$master_port \
      ${torchrun_args[*]} 2>&1 | tee '$log_file'
  "
}

if [[ "$RUN_NONUNIFORM" == "1" ]]; then
  echo "=== nonuniform NEP EP32/EP28 TP2 CP2 ETP1 ==="
  make_profile_args nonuniform
  if ! run_in_allocation 15 "$LOGDIR/nonuniform_ep32_ep28_tp2_cp2.log" "$MASTER_PORT" \
    examples/nonuniform/pretrain_gpt_nonuniform.py "${common_args[@]}" "${PROFILE_ARGS[@]}" \
    --global-batch-size "$NONUNIFORM_GBS" \
    --nonuniform-mode ep \
    --nonuniform-ep-num-tp-cp-per-replica 8 7; then
    status=1
  fi
fi

if [[ "$RUN_UNIFORM" == "1" ]]; then
  echo "=== uniform EP32 TP2 CP2 ETP1 standard ==="
  make_profile_args uniform
  if ! run_in_allocation 16 "$LOGDIR/uniform_ep32_tp2_cp2.log" "$((MASTER_PORT + 1))" \
    pretrain_gpt.py "${common_args[@]}" "${PROFILE_ARGS[@]}" \
    --global-batch-size "$UNIFORM_GBS" \
    --expert-model-parallel-size 32; then
    status=1
  fi
fi

echo "Logs written to $LOGDIR"
exit "$status"
