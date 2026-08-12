#!/bin/bash

# Original a3b/30b hybrid model with 128 true experts on NEP EP16/EP12.
# Flex uses HybridEP virtual slots; optimizer stepping is skipped for this full-size benchmark.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b128e-ep16-12-flex-noopt

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_30b_original_128e_ep16_ep12_flex_no_optimizer}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-15m}"
TRAIN_ITERS="${TRAIN_ITERS:-4}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-3}"
PROFILE_RANKS="${PROFILE_RANKS:-0 16}"
NAME="a3b_30b_original_128e_ep16_ep12_flex_hybridep_noopt_i${TRAIN_ITERS}"
RUN_DIR="${ROOT_DIR}/${NAME}/${SLURM_JOB_ID}"
DRIVER_LOG="${RUN_DIR}/driver_${SLURM_JOB_ID}.log"
HYBRID_LAYER_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

mkdir -p "${REPO_DIR}/slurm_runs/lyris" "${RUN_DIR}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 8)); then
    echo "Expected eight allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
RUN_NODELIST=$(IFS=,; echo "${allocated_nodes[*]}")
MASTER_ADDR="${allocated_nodes[0]}"

# Warm the named Enroot cache on every node, then perform one scoped runtime and
# EP16/EP12 placement preflight before consuming the 28 training GPUs.
srun --nodes=8 --nodelist="${RUN_NODELIST}" --ntasks=8 --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc 'python -c "import torch, mamba_ssm, causal_conv1d; print(torch.__version__)"'

srun --nodes=1 --nodelist="${MASTER_ADDR}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${REPO_DIR}'
python -m py_compile \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    pretrain_hybrid.py
python -c '
from megatron.core.distributed.nonuniform_common import (
    compute_nonuniform_ep_dispatch_slots,
    compute_nonuniform_ep_expert_placement,
    compute_nonuniform_ep_owner_expert_slots,
)
from megatron.core.transformer.moe.token_dispatcher import (
    _pad_nonuniform_flex_dispatch_slots,
)
owner = compute_nonuniform_ep_owner_expert_slots(128, 12)
full, _ = compute_nonuniform_ep_expert_placement(128, 16, 12)
reduced, _ = compute_nonuniform_ep_expert_placement(128, 12, 12)
assert [sum(expert is not None for expert in row) for row in owner] == [11] * 8 + [10] * 4
assert [len(row) for row in full] == [8] * 16
assert [len(row) for row in reduced] == [11] * 8 + [10] * 4
assert sorted(expert for row in full for expert in row) == list(range(128))
assert sorted(expert for row in reduced for expert in row) == list(range(128))
full_slots = _pad_nonuniform_flex_dispatch_slots(
    compute_nonuniform_ep_dispatch_slots(full, 128), 16, \"hybridep\"
)
reduced_slots = _pad_nonuniform_flex_dispatch_slots(
    compute_nonuniform_ep_dispatch_slots(reduced, 128), 12, \"hybridep\"
)
assert len(full_slots[0]) == 8
assert len(reduced_slots[0]) == 11
assert sum(expert is None for row in reduced_slots for expert in row) == 4
print(\"[a3b-ep16-12-flex-noopt] placement preflight: PASS\")
'
"

echo "[a3b-ep16-12-flex-noopt] $(date --iso-8601=seconds) starting ${NAME}"
timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        NAME="${NAME}" \
        MASTER_ADDR="${MASTER_ADDR}" \
        MASTER_PORT=30961 \
        RUN_NNODES=7 \
        RUN_WORLD_SIZE=28 \
        GPUS_PER_NODE=4 \
        NVLINK_SEGMENT_NODES=4 \
        USE_DIRECT_SRUN_RANKS=1 \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        LR_WSD_DECAY_ITERS=2 \
        HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN}" \
        NUM_EXPERTS=128 \
        MOE_ROUTER_TOPK=6 \
        SEQ_LENGTH=8192 \
        NONUNIFORM_MODE=ep \
        NONUNIFORM_EP_TOPOLOGY="8 6" \
        TENSOR_MODEL_PARALLEL_SIZE=2 \
        CONTEXT_PARALLEL_SIZE=1 \
        EXPERT_MODEL_PARALLEL_SIZE=16 \
        EXPERT_TENSOR_PARALLEL_SIZE=1 \
        MICRO_BATCH_SIZE=1 \
        GLOBAL_BATCH_SIZE=14 \
        TRUE_GLOBAL_BATCH_SIZE=14 \
        REPLICA_MICRO_BATCH_SIZES="1 1" \
        REPLICA_NUM_MICROBATCHES="1 1" \
        DDP_NUM_BUCKETS=8 \
        CUDA_GRAPH_IMPL=none \
        PROFILE=1 \
        PROFILE_STEP_START="${PROFILE_STEP_START}" \
        PROFILE_STEP_END="${PROFILE_STEP_END}" \
        PROFILE_RANKS="${PROFILE_RANKS}" \
        LOG_PARAMS_NORM=0 \
        LOG_NUM_ZEROS_IN_GRAD=0 \
        LOG_ENERGY=0 \
        MANUAL_GC_INTERVAL=1000 \
        LOG_INTERVAL=1 \
        EXTRA_MEGATRON_ARGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --calculate-per-token-loss --recompute-granularity full --recompute-method uniform --recompute-num-layers 1" \
        HIGH_PRIORITY_STREAM_GROUPS=ep \
        CUDA_DEVICE_MAX_CONNECTIONS=32 \
        NCCL_LAUNCH_ORDER_IMPLICIT=1 \
        TORCH_NCCL_BLOCKING_WAIT=0 \
        NCCL_DEBUG=WARN \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_UNIFORM_ROUTING=0 \
        MOE_ROUTER_ENABLE_EXPERT_BIAS=1 \
        NONUNIFORM_SKIP_OPTIMIZER_STEP=1 \
        MEGATRON_NONUNIFORM_EP_DEBUG=1 \
        MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="0 16" \
        MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
        MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
        MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 \
        MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1 \
        MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1 \
        MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0 \
        MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 \
        MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0 \
        MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1 \
        MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0 \
        MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1 \
        MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW=1 \
        MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 \
        MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1 \
        MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
        MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0 \
        MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
        MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1 \
        MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64 \
        MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=3 \
        MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 \
        MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0 \
        MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=0 \
        MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter \
        DISTRIBUTED_TIMEOUT_MINUTES=5 \
        EXIT_DURATION_IN_MINS=14 \
        USE_GLOO_PROCESS_GROUPS=1 \
        SKIP_PREFLIGHT=1 \
        bash "${RUNNER}"

echo "[a3b-ep16-12-flex-noopt] $(date --iso-8601=seconds) training completed"

python3 - "${DRIVER_LOG}" "${RUN_DIR}/torch_profile" "${TRAIN_ITERS}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
profile_dir = Path(sys.argv[2])
expected_iters = int(sys.argv[3])
text = log_path.read_text()

required_config = {
    "Flex token dispatcher": r"moe_token_dispatcher_type\s+\.+\s+flex",
    "HybridEP backend": r"moe_flex_dispatcher_backend\s+\.+\s+hybridep",
    "non-distributed optimizer mode": r"use_distributed_optimizer\s+\.+\s+False",
    "optimizer step skipped": r"nonuniform_skip_optimizer_step=1",
    "128 true experts": r"num_experts\s+\.+\s+128",
}
for label, pattern in required_config.items():
    if re.search(pattern, text) is None:
        raise RuntimeError(f"Missing runtime confirmation for {label}")

iteration_pattern = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?"
    r"number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)
records = [tuple(map(int, match.groups())) for match in iteration_pattern.finditer(text)]
if [record[0] for record in records[-expected_iters:]] != list(range(1, expected_iters + 1)):
    raise RuntimeError(f"Missing training iterations: {records}")
if any(
    total != expected_iters or skipped != 0 or nan != 0
    for _, total, skipped, nan in records[-expected_iters:]
):
    raise RuntimeError(f"Invalid training iteration status: {records[-expected_iters:]}")

for rank in (0, 16):
    trace = profile_dir / f"rank-{rank}.json.gz"
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(f"Missing profiler trace: {trace}")

print(
    "[a3b-ep16-12-flex-noopt] PASS: original 128-expert a3b/30b architecture, "
    "EP16/EP12, Flex/HybridEP, no optimizer step, "
    f"{expected_iters} finite iterations, and profiles"
)
PY
