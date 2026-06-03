#!/bin/bash
set -euo pipefail

MASTER_ADDR_ARG=$1
MASTER_PORT_ARG=$2
LOGDIR_ARG=$3
GPUS_PER_NODE_ARG=$4

cd "$(dirname "$0")/.."

ulimit -s 8192
if [[ -n "${DEEP_EP_LD_PRELOAD:-}" && -f "$DEEP_EP_LD_PRELOAD" ]]; then
  export LD_PRELOAD="$DEEP_EP_LD_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi

python - <<'PY'
import importlib.util
import torch

names = ("pandas", "deep_ep", "nvshmem.core", "megatron")
ok = {name: importlib.util.find_spec(name) is not None for name in names}
print(ok)
missing = [name for name, present in ok.items() if not present]
if missing:
    raise SystemExit(f"missing dependencies: {missing}")

cuda_count = torch.cuda.device_count()
print("cuda_count", cuda_count)
if cuda_count != 4:
    raise SystemExit(f"expected 4 visible GPUs, got {cuda_count}")
PY

capacity_args=()
if [[ -n "${MOE_EXPERT_CAPACITY_FACTOR:-}" ]]; then
  capacity_args+=(--moe-expert-capacity-factor "$MOE_EXPERT_CAPACITY_FACTOR")
fi
if [[ "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-0}" == "1" ]]; then
  capacity_args+=(--moe-pad-expert-input-to-capacity)
fi

tp_size=${TP_SIZE:-1}
ep_size=${EP_SIZE:-4}
tp_overlap_args=()
if [[ -n "${TP_COMM_OVERLAP_CFG:-}" ]]; then
  tp_overlap_args+=(
    --sequence-parallel
    --tp-comm-overlap
    --tp-comm-overlap-cfg "$TP_COMM_OVERLAP_CFG"
  )
fi

torchrun --nproc_per_node="$GPUS_PER_NODE_ARG" --nnodes=1 \
  --node_rank=0 --master_addr="$MASTER_ADDR_ARG" --master_port="$MASTER_PORT_ARG" \
  pretrain_gpt.py \
  --mock-data \
  --tokenizer-type NullTokenizer \
  --vocab-size 4096 \
  --train-iters 1 \
  --eval-iters 0 \
  --eval-interval 1 \
  --log-interval 1 \
  --no-check-for-nan-in-loss-and-grad \
  --ddp-average-in-collective \
  --bf16 \
  --grad-reduce-in-bf16 \
  --tensor-model-parallel-size "$tp_size" \
  --context-parallel-size 1 \
  --expert-model-parallel-size "$ep_size" \
  --expert-tensor-parallel-size 1 \
  --num-layers 1 \
  --hidden-size 512 \
  --ffn-hidden-size 2048 \
  --num-attention-heads 8 \
  --seq-length 128 \
  --max-position-embeddings 128 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --num-experts 4 \
  --moe-router-topk 2 \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 0.01 \
  --moe-token-dispatcher-type flex \
  --moe-flex-dispatcher-backend hybridep \
  --moe-router-dtype fp32 \
  --moe-grouped-gemm \
  --moe-permute-fusion \
  --moe-router-fusion \
  --moe-router-force-load-balancing \
  "${capacity_args[@]}" \
  "${tp_overlap_args[@]}" \
  --disable-bias-linear \
  --transformer-impl transformer_engine \
  --lr 1.0e-4 \
  --min-lr 1.0e-5 \
  --lr-decay-style constant \
  2>&1 | tee "$LOGDIR_ARG/smoke.log"
