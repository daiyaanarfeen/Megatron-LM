#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=8
#SBATCH --time=00:30:00
#SBATCH --job-name=std_train_ep8_6
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Standard Megatron GPT training comparison:
# - heterogeneous EP8/EP6, TP2, CP2, ETP2 on 28 ranks
# - uniform EP8/EP8, TP2, CP2, ETP2 on 32 ranks
# Runs each heterogeneous EP DDP opt-in approach for the nonuniform case:
# nccl, nvshmem, phased. The uniform baseline runs once because it does not
# exercise a nonuniform reshard path.
#
# In Slurm, request at least 8 nodes with 4 GPUs per node, then run this script
# inside the allocation. Set IMAGE to use the same containerized srun pattern as
# the other heterogeneous EP scripts, or run without IMAGE if the environment
# already has Megatron dependencies installed.

set -euo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
LOGDIR=${LOGDIR:-$WORKDIR/heterogeneous_ep_training_logs}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TRAIN_ITERS=${TRAIN_ITERS:-5}
NVSHMEM_SLOT_MB=${MEGATRON_NVSHMEM_SLOT_MB:-256}
RUN_HETERO=${RUN_HETERO:-1}
RUN_UNIFORM=${RUN_UNIFORM:-1}
HETERO_APPROACHES=${HETERO_APPROACHES:-"nccl nvshmem phased"}
HSG_CACHED_IMAGE=${HSG_CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-nvshmem-megatron-het-ep.sqsh}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVSHMEM_MAX_TEAMS=${NVSHMEM_MAX_TEAMS:-512}
export NVSHMEM_DISABLE_NVLS=${NVSHMEM_DISABLE_NVLS:-1}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4G}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}

if [[ -z "${IMAGE:-}" && -f "$HSG_CACHED_IMAGE" ]]; then
  IMAGE=$HSG_CACHED_IMAGE
fi

if [[ -z "${INSTALL_NVSHMEM+x}" ]]; then
  if [[ "${IMAGE:-}" == "$HSG_CACHED_IMAGE" ]]; then
    INSTALL_NVSHMEM=0
  else
    INSTALL_NVSHMEM=1
  fi
fi

mkdir -p "$LOGDIR"

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
  --bf16
  --tensor-model-parallel-size 2
  --sequence-parallel
  --context-parallel-size 2
  --expert-tensor-parallel-size 2
  --num-layers 2
  --hidden-size 128
  --ffn-hidden-size 512
  --num-attention-heads 8
  --seq-length 128
  --max-position-embeddings 128
  --micro-batch-size 1
  --num-experts 24
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

run_torchrun() {
  local label=$1
  local nnodes=$2
  local global_batch=$3
  local approach=$4
  shift 4
  local topology_args=("$@")
  local log_file="$LOGDIR/${label}_${approach}.log"

  echo "=== $label approach=$approach nnodes=$nnodes global_batch=$global_batch ==="

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local master
    master=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    master=$(getent hosts "$master" | awk '{print $1}' | head -n1)
    master=${master:-$(scontrol show hostname "$SLURM_NODELIST" | head -n1)}

    local srun_prefix=(
      srun --nodes="$nnodes" --ntasks="$nnodes" --ntasks-per-node=1
    )
    if [[ -n "${IMAGE:-}" ]]; then
      srun_prefix+=(
        --container-image="$IMAGE"
        --container-mounts="$WORKDIR:$WORKDIR"
        --container-workdir="$WORKDIR"
        --container-env=NCCL_NVLS_ENABLE,NVSHMEM_MAX_TEAMS,NVSHMEM_DISABLE_NVLS
        --container-env=NCCL_LAUNCH_ORDER_IMPLICIT,NVSHMEM_SYMMETRIC_SIZE,UCX_NET_DEVICES
      )
    fi

    "${srun_prefix[@]}" bash -lc "
      set -euo pipefail
      ulimit -s 8192
      export MEGATRON_NVSHMEM_SLOT_MB=$NVSHMEM_SLOT_MB
      if [[ '$approach' == 'nvshmem' && $INSTALL_NVSHMEM == 1 ]]; then
        pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null || true
        echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib \
          > /etc/ld.so.conf.d/nvshmem.conf 2>/dev/null || true
        ldconfig 2>/dev/null || true
      fi
      cd '$WORKDIR'
      torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$nnodes \
        --node_rank=\$SLURM_NODEID --master_addr=$master --master_port=$MASTER_PORT \
        pretrain_gpt_heterogeneous_ep.py ${common_args[*]} --global-batch-size $global_batch \
        --heterogeneous-ep-ddp-approach $approach \
        ${topology_args[*]} 2>&1 | tee '$log_file'
    "
  else
    torchrun --nproc_per_node="$GPUS_PER_NODE" --nnodes="$nnodes" \
      --master_port="$MASTER_PORT" \
      pretrain_gpt_heterogeneous_ep.py "${common_args[@]}" --global-batch-size "$global_batch" \
      --heterogeneous-ep-ddp-approach "$approach" \
      "${topology_args[@]}" 2>&1 | tee "$log_file"
  fi
}

run_uniform_baseline() {
  local label=$1
  local nnodes=$2
  local global_batch=$3
  local log_file="$LOGDIR/${label}.log"

  echo "=== $label standard uniform EP8 nnodes=$nnodes global_batch=$global_batch ==="

  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    local master
    master=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    master=$(getent hosts "$master" | awk '{print $1}' | head -n1)
    master=${master:-$(scontrol show hostname "$SLURM_NODELIST" | head -n1)}

    local srun_prefix=(
      srun --nodes="$nnodes" --ntasks="$nnodes" --ntasks-per-node=1
    )
    if [[ -n "${IMAGE:-}" ]]; then
      srun_prefix+=(
        --container-image="$IMAGE"
        --container-mounts="$WORKDIR:$WORKDIR"
        --container-workdir="$WORKDIR"
        --container-env=NCCL_NVLS_ENABLE,NVSHMEM_MAX_TEAMS,NVSHMEM_DISABLE_NVLS
        --container-env=NCCL_LAUNCH_ORDER_IMPLICIT,NVSHMEM_SYMMETRIC_SIZE,UCX_NET_DEVICES
      )
    fi

    "${srun_prefix[@]}" bash -lc "
      set -euo pipefail
      ulimit -s 8192
      cd '$WORKDIR'
      torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$nnodes \
        --node_rank=\$SLURM_NODEID --master_addr=$master --master_port=$MASTER_PORT \
        pretrain_gpt.py ${common_args[*]} --global-batch-size $global_batch \
        --expert-model-parallel-size 8 2>&1 | tee '$log_file'
    "
  else
    torchrun --nproc_per_node="$GPUS_PER_NODE" --nnodes="$nnodes" \
      --master_port="$MASTER_PORT" \
      pretrain_gpt.py "${common_args[@]}" --global-batch-size "$global_batch" \
      --expert-model-parallel-size 8 2>&1 | tee "$log_file"
  fi
}

if [[ "$RUN_HETERO" == "1" ]]; then
  for approach in $HETERO_APPROACHES; do
    run_torchrun "hetero_ep8_ep6_tp2_cp2" 7 7 "$approach" \
      --heterogeneous-ep-num-tp-cp-per-replica 4 3
  done
fi

if [[ "$RUN_UNIFORM" == "1" ]]; then
  run_uniform_baseline "uniform_ep8_tp2_cp2_standard" 8 8
fi

echo "Logs written to $LOGDIR"
