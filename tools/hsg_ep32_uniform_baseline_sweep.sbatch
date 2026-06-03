#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=16
#SBATCH --time=00:45:00
#SBATCH --job-name=ep32_base_sweep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port}
BASE_LOG_ROOT=${BASE_LOG_ROOT:-$WORKDIR/ep32_uniform_baseline_sweep_$(date +%Y%m%d_%H%M%S)}
TRAIN_ITERS=${TRAIN_ITERS:-8}
PROFILE=${PROFILE:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
HSG_CACHED_IMAGE=${HSG_CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-deepep-hybridep-megatron-nep-v3.sqsh}
ENROOT_ROOT=${ENROOT_ROOT:-/lustre/fsw/portfolios/coreai/users/darfeen/enroot}
ENROOT_CONTAINER_NAME=${ENROOT_CONTAINER_NAME:-megatron_pyt2510_hybridep_v3}

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$ENROOT_ROOT/cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$ENROOT_ROOT/images}
export ENROOT_TEMP_PATH=${ENROOT_TEMP_PATH:-/tmp}
if [[ "${PYTORCH_ALLOC_CONF:-}" == "unset" || "${PYTORCH_ALLOC_CONF:-}" == "none" || "${PYTORCH_ALLOC_CONF:-}" == "off" ]]; then
  unset PYTORCH_ALLOC_CONF
else
  export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
fi
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == "unset" || "${PYTORCH_CUDA_ALLOC_CONF:-}" == "none" || "${PYTORCH_CUDA_ALLOC_CONF:-}" == "off" ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
else
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
fi
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}
export EP_NIC_NAME=${EP_NIC_NAME:-mlx5_0}
export EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-$WORKDIR/.cache/deep_ep_jit}
export DEEP_EP_LD_PRELOAD=${DEEP_EP_LD_PRELOAD:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3}

cd "$WORKDIR"
mkdir -p "$BASE_LOG_ROOT" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

should_run_case() {
  local name=$1
  local run_cases=${RUN_CASES:-all}
  run_cases=${run_cases//,/ }
  run_cases=${run_cases//:/ }
  [[ "$run_cases" == "all" || " $run_cases " == *" $name "* ]]
}

run_case() {
  local name=$1
  local port=$2
  shift 2

  if ! should_run_case "$name"; then
    echo "=== skipping uniform EP32 baseline sweep: $name ==="
    return 0
  fi

  echo "=== uniform EP32 baseline sweep: $name ==="
  (
    export "$@"
    export CUDA_DEVICE_MAX_CONNECTIONS
    local logdir="$BASE_LOG_ROOT/$name"
    local log_file="$logdir/uniform_ep32_tp2_cp2.log"
    mkdir -p "$logdir"

    local args=(
      --mock-data
      --tokenizer-type NullTokenizer
      --vocab-size 4096
      --train-iters "$TRAIN_ITERS"
      --eval-iters 0
      --eval-interval "$TRAIN_ITERS"
      --log-interval 1
      --timing-log-level 1
      --timing-log-option minmax
      --log-throughput
      --no-check-for-nan-in-loss-and-grad
      --ddp-average-in-collective
      --bf16
      --grad-reduce-in-bf16
      --tensor-model-parallel-size 2
      --context-parallel-size 2
      --expert-tensor-parallel-size 1
      --sequence-parallel
      --num-layers "$NUM_LAYERS"
      --hidden-size "$HIDDEN_SIZE"
      --ffn-hidden-size "$FFN_HIDDEN_SIZE"
      --num-attention-heads "${NUM_ATTENTION_HEADS:-32}"
      --seq-length "$SEQ_LENGTH"
      --max-position-embeddings "${MAX_POSITION_EMBEDDINGS:-$SEQ_LENGTH}"
      --micro-batch-size "${MICRO_BATCH_SIZE:-1}"
      --num-experts "$NUM_EXPERTS"
      --moe-router-topk 2
      --moe-router-load-balancing-type aux_loss
      --moe-aux-loss-coeff 0.01
      --moe-token-dispatcher-type "$MOE_TOKEN_DISPATCHER_TYPE"
      --moe-grouped-gemm
      --disable-bias-linear
      --transformer-impl transformer_engine
      --lr 1.0e-4
      --min-lr 1.0e-5
      --lr-decay-style constant
      --global-batch-size "$UNIFORM_GBS"
      --expert-model-parallel-size 32
    )

    if [[ "${OVERLAP_GRAD_REDUCE:-1}" == "1" ]]; then
      args+=(--overlap-grad-reduce)
    fi
    if [[ "${USE_DISTRIBUTED_OPTIMIZER:-0}" == "1" ]]; then
      args+=(--use-distributed-optimizer)
    fi
    if [[ "${OVERLAP_PARAM_GATHER:-0}" == "1" ]]; then
      args+=(--overlap-param-gather)
    fi
    if [[ "${USE_NCCL_UB:-0}" == "1" ]]; then
      args+=(--use-nccl-ub)
    fi
    if [[ "${DDP_PAD_BUCKETS_FOR_HIGH_NCCL_BUSBW:-0}" == "1" ]]; then
      args+=(--ddp-pad-buckets-for-high-nccl-busbw)
    fi
    if [[ "${TP_COMM_OVERLAP:-1}" == "1" ]]; then
      args+=(--tp-comm-overlap)
    fi
    if [[ -n "${TP_COMM_OVERLAP_CFG:-}" ]]; then
      args+=(--tp-comm-overlap-cfg "$TP_COMM_OVERLAP_CFG")
    fi
    if [[ "${TP_COMM_OVERLAP_RS_DGRAD:-1}" == "1" ]]; then
      args+=(--tp-comm-overlap-rs-dgrad)
    fi
    if [[ "${TP_COMM_BULK_WGRAD:-0}" == "0" ]]; then
      args+=(--no-tp-comm-bulk-wgrad)
    fi
    if [[ "${OVERLAP_MOE_EXPERT_PARALLEL_COMM:-1}" == "1" ]]; then
      args+=(--overlap-moe-expert-parallel-comm)
    fi
    if [[ "${DELAY_WGRAD_COMPUTE:-1}" == "1" ]]; then
      args+=(--delay-wgrad-compute)
    fi
    if [[ "${MOE_PERMUTE_FUSION:-1}" == "1" ]]; then
      args+=(--moe-permute-fusion)
    fi
    if [[ "${MOE_ROUTER_FUSION:-1}" == "1" ]]; then
      args+=(--moe-router-fusion)
    fi
    if [[ "${MOE_ROUTER_FORCE_LOAD_BALANCING:-1}" == "1" ]]; then
      args+=(--moe-router-force-load-balancing)
    fi
    if [[ "$MOE_TOKEN_DISPATCHER_TYPE" == "flex" ]]; then
      args+=(
        --moe-flex-dispatcher-backend "$MOE_FLEX_DISPATCHER_BACKEND"
        --moe-router-dtype "${MOE_ROUTER_DTYPE:-fp32}"
      )
    fi
    if [[ -n "${MOE_HYBRIDEP_NUM_SMS:-}" ]]; then
      args+=(--moe-hybridep-num-sms "$MOE_HYBRIDEP_NUM_SMS")
    fi
    if [[ -n "${MOE_EXPERT_CAPACITY_FACTOR:-}" ]]; then
      args+=(--moe-expert-capacity-factor "$MOE_EXPERT_CAPACITY_FACTOR")
    fi
    if [[ "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-0}" == "1" ]]; then
      args+=(--moe-pad-expert-input-to-capacity)
    fi
    if [[ -n "${DDP_BUCKET_SIZE:-}" ]]; then
      args+=(--ddp-bucket-size "$DDP_BUCKET_SIZE")
    fi
    local recompute_granularity="${RECOMPUTE_GRANULARITY:-selective}"
    local recompute_modules_value="${RECOMPUTE_MODULES:-core_attn}"
    if [[ "$recompute_granularity" != "none" && "$recompute_granularity" != "off" ]]; then
      args+=(--recompute-granularity "$recompute_granularity")
      if [[ -n "$recompute_modules_value" && "$recompute_modules_value" != "none" && "$recompute_modules_value" != "off" ]]; then
        read -r -a recompute_modules <<< "$recompute_modules_value"
        args+=(--recompute-modules "${recompute_modules[@]}")
      fi
    fi
    if [[ -n "${CP_COMM_TYPE:-p2p}" ]]; then
      read -r -a cp_comm_type_args <<< "${CP_COMM_TYPE:-p2p}"
      args+=(--cp-comm-type "${cp_comm_type_args[@]}")
    fi
    if [[ "${DISABLE_GLOO_PROCESS_GROUPS:-1}" == "1" ]]; then
      args+=(--disable-gloo-process-groups)
    fi
    if [[ "$PROFILE" == "1" ]]; then
      read -r -a profile_ranks <<< "${PROFILE_RANKS:-0 28 56}"
      args+=(
        --profile
        --use-pytorch-profiler
        --profile-step-start "${PROFILE_STEP_START:-5}"
        --profile-step-end "${PROFILE_STEP_END:-7}"
        --profile-ranks "${profile_ranks[@]}"
        --tensorboard-dir "$logdir/uniform/tensorboard"
      )
    fi

    local master
    master=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
    master=$(getent hosts "$master" | awk '{print $1}' | head -n1)
    master=${master:-$(scontrol show hostname "$SLURM_NODELIST" | head -n1)}

    local srun_cmd=(srun --nodes=16 --ntasks=16 --ntasks-per-node=1 --export=ALL)

    "${srun_cmd[@]}" bash -lc "
      set -euo pipefail
      export ENROOT_CACHE_PATH='$ENROOT_CACHE_PATH'
      export ENROOT_DATA_PATH='$ENROOT_DATA_PATH'
      export ENROOT_TEMP_PATH='$ENROOT_TEMP_PATH'
      export NVIDIA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\"
      export NV_GPU=\"\$CUDA_VISIBLE_DEVICES\"
      export NVIDIA_DRIVER_CAPABILITIES=all
      node_container='${ENROOT_CONTAINER_NAME}_node'\$SLURM_NODEID
      if ! enroot list | awk '{print \$1}' | grep -Fxq \"\$node_container\"; then
        echo \"Creating cached Enroot container: \$node_container from $HSG_CACHED_IMAGE\"
        enroot create -n \"\$node_container\" '$HSG_CACHED_IMAGE'
      else
        echo \"Using cached Enroot container: \$node_container\"
      fi
      enroot start \
        --mount '$WORKDIR:$WORKDIR' \
        --env CUDA_VISIBLE_DEVICES \
        --env NVIDIA_VISIBLE_DEVICES \
        --env NV_GPU \
        --env NVIDIA_DRIVER_CAPABILITIES \
        --env SLURM_NODEID \
        --env NCCL_NVLS_ENABLE \
        --env NCCL_LAUNCH_ORDER_IMPLICIT \
        --env CUDA_DEVICE_MAX_CONNECTIONS \
        --env UCX_NET_DEVICES \
        --env PYTORCH_ALLOC_CONF \
        --env PYTORCH_CUDA_ALLOC_CONF \
        --env EP_NIC_NAME \
        --env EP_JIT_CACHE_DIR \
        --env DEEP_EP_LD_PRELOAD \
        \"\$node_container\" bash -lc '
          set -euo pipefail
          ulimit -s 8192
          if [[ -n \"\${DEEP_EP_LD_PRELOAD:-}\" && -f \"\$DEEP_EP_LD_PRELOAD\" ]]; then
            export LD_PRELOAD=\"\$DEEP_EP_LD_PRELOAD\${LD_PRELOAD:+:\$LD_PRELOAD}\"
          fi
          cd '$WORKDIR'
          torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=16 \
            --node_rank=\$SLURM_NODEID --master_addr=$master --master_port=$port \
            pretrain_gpt.py ${args[*]} 2>&1 | tee '$log_file'
        '
    "
  )
}

# Reference shape from the latest EP32 HybridEP run, but with CMD=1 to test the
# TP/SP scheduling recommendation directly against the previous CMD=32 result.
run_case current_cmd1 29720 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=2 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=65536 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

# Same total expert FFN width as the previous run, spread across more layers.
# This should give earlier bucket readiness and more later backward compute to
# hide expert DP allreduce, without increasing expert parameters materially.
run_case deeper_l4_cmd32 29721 \
  CUDA_DEVICE_MAX_CONNECTIONS=32 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=4 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=32768 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

# Same deeper shape with CMD=1 to expose whether TP/SP overlap wins more than
# HybridEP overlap loses on this hardware/software stack.
run_case deeper_l4_cmd1 29722 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=4 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=32768 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

run_case ub_ref_cmd1 29723 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=2 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=65536 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_CFG=$WORKDIR/tools/hsg_tp2_ub_ref.yaml \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

run_case ub_low_sm_cmd1 29724 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=2 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=65536 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_CFG=$WORKDIR/tools/hsg_tp2_ub_low_sm.yaml \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

run_case ub_ring_bulk_cmd1 29725 \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  UNIFORM_GBS=512 \
  SEQ_LENGTH=32768 \
  NUM_LAYERS=2 \
  HIDDEN_SIZE=4096 \
  FFN_HIDDEN_SIZE=65536 \
  NUM_EXPERTS=224 \
  MOE_TOKEN_DISPATCHER_TYPE=flex \
  MOE_FLEX_DISPATCHER_BACKEND=hybridep \
  MOE_HYBRIDEP_NUM_SMS=16 \
  MOE_PERMUTE_FUSION=1 \
  MOE_ROUTER_FUSION=1 \
  MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
  DDP_BUCKET_SIZE=1500000000 \
  OVERLAP_GRAD_REDUCE=1 \
  TP_COMM_OVERLAP=1 \
  TP_COMM_OVERLAP_CFG=$WORKDIR/tools/hsg_tp2_ub_ring_bulk.yaml \
  TP_COMM_OVERLAP_RS_DGRAD=1 \
  TP_COMM_BULK_WGRAD=0 \
  OVERLAP_MOE_EXPERT_PARALLEL_COMM=1 \
  DELAY_WGRAD_COMPUTE=1

echo "Logs written under $BASE_LOG_ROOT"
