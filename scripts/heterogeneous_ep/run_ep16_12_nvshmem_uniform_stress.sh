#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=8
#SBATCH --time=01:00:00
#SBATCH --job-name=ep16_12_stress
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Standard Megatron GPT stress comparison:
# - heterogeneous EP16/EP12, TP2, CP2, ETP1 on 28 ranks
# - uniform EP16, TP2, CP2, ETP1 on 32 ranks, standard pretrain_gpt.py
# - proportional per-replica samples: uniform 16-GPU replicas get 8 samples,
#   hetero 16-GPU replica gets 8 and 12-GPU replica gets 6.

set -euo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
LOGDIR=${LOGDIR:-$WORKDIR/heterogeneous_ep_training_logs_ep16_12_stress}
MASTER_PORT=${MASTER_PORT:-29520}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAIN_ITERS=${TRAIN_ITERS:-20}
HETERO_APPROACHES=${HETERO_APPROACHES:-"nvshmem"}
HETERO_GBS=${HETERO_GBS:-14}
UNIFORM_GBS=${UNIFORM_GBS:-16}
SEQ_LENGTH=${SEQ_LENGTH:-8192}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
NUM_LAYERS=${NUM_LAYERS:-4}
HIDDEN_SIZE=${HIDDEN_SIZE:-4096}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-16384}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-32}
NUM_EXPERTS=${NUM_EXPERTS:-96}
# The opt-in NVSHMEM path rebuilds expert buckets on layer boundaries. For this
# model each layer/expert BF16 grad payload is 256 MiB, so 320 MiB leaves margin.
NVSHMEM_SLOT_MB=${MEGATRON_NVSHMEM_SLOT_MB:-320}
RUN_HETERO=${RUN_HETERO:-1}
RUN_UNIFORM=${RUN_UNIFORM:-1}
HSG_CACHED_IMAGE=${HSG_CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-nvshmem-megatron-het-ep.sqsh}

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVSHMEM_MAX_TEAMS=${NVSHMEM_MAX_TEAMS:-512}
export NVSHMEM_DISABLE_NVLS=${NVSHMEM_DISABLE_NVLS:-1}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-96G}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

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
  --micro-batch-size 1
  --num-experts "$NUM_EXPERTS"
  --moe-router-topk 2
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 0.01
  --moe-token-dispatcher-type alltoall
  --disable-bias-linear
  --transformer-impl transformer_engine
  --lr 1.0e-4
  --min-lr 1.0e-5
  --lr-decay-style constant
)

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
      --container-env=NCCL_NVLS_ENABLE,NVSHMEM_MAX_TEAMS,NVSHMEM_DISABLE_NVLS
      --container-env=NCCL_LAUNCH_ORDER_IMPLICIT,NVSHMEM_SYMMETRIC_SIZE,UCX_NET_DEVICES
      --container-env=CUDA_DEVICE_MAX_CONNECTIONS
    )
  fi

  "${srun_cmd[@]}" bash -lc "
    set -euo pipefail
    ulimit -s 8192
    export MEGATRON_NVSHMEM_SLOT_MB=$NVSHMEM_SLOT_MB
    cd '$WORKDIR'
    torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$nnodes \
      --node_rank=\$SLURM_NODEID --master_addr=$master --master_port=$master_port \
      ${torchrun_args[*]} 2>&1 | tee '$log_file'
  "
}

if [[ "$RUN_HETERO" == "1" ]]; then
  approach_index=0
  for approach in $HETERO_APPROACHES; do
    echo "=== hetero EP16/EP12 TP2 CP2 ETP1 approach=$approach ==="
    if ! run_in_allocation 7 "$LOGDIR/hetero_ep16_ep12_tp2_cp2_${approach}.log" "$((MASTER_PORT + approach_index))" \
      pretrain_gpt_heterogeneous_ep.py "${common_args[@]}" \
      --global-batch-size "$HETERO_GBS" \
      --heterogeneous-ep-ddp-approach "$approach" \
      --heterogeneous-ep-num-tp-cp-per-replica 4 3; then
      status=1
    fi
    approach_index=$((approach_index + 1))
  done
else
  approach_index=0
fi

if [[ "$RUN_UNIFORM" == "1" ]]; then
  echo "=== uniform EP16 TP2 CP2 ETP1 standard ==="
  if ! run_in_allocation 8 "$LOGDIR/uniform_ep16_tp2_cp2_standard.log" "$((MASTER_PORT + approach_index))" \
    pretrain_gpt.py "${common_args[@]}" \
    --global-batch-size "$UNIFORM_GBS" \
    --expert-model-parallel-size 16; then
    status=1
  fi
fi

echo "Logs written to $LOGDIR"
exit "$status"
