#!/bin/bash

# Same-allocation comparison of the original a3b/30b workload with native
# distributed optimizer support:
#   healthy: EP16 + EP16 (32 ranks)
#   NEP:     EP16 + EP12 (28 ranks, nonuniform placement for 128 experts)
#
# Iterations 1-4 warm up, 5-7 cover the profiler window, and 8-10 are
# profiler-free timing samples.  The two cases run sequentially on the same
# allocation, image, segment layout, and software checkout.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b-ep16-12-distopt-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_128e_ep16_ep12_distopt_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TRAIN_ITERS="${TRAIN_ITERS:-10}"
PROFILE_STEP_START="${PROFILE_STEP_START:-5}"
PROFILE_STEP_END="${PROFILE_STEP_END:-7}"
CASE_TIMEOUT="${CASE_TIMEOUT:-17m}"
RUN_HEALTHY_CASE="${RUN_HEALTHY_CASE:-1}"
RUN_NEP_CASE="${RUN_NEP_CASE:-1}"
NEP_DEBUG_RANKS="${NEP_DEBUG_RANKS:-0 16}"
MOE_EXPERT_RANK_CAPACITY_FACTOR="${MOE_EXPERT_RANK_CAPACITY_FACTOR:-1.5}"
HYBRID_LAYER_PATTERN="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

mkdir -p "${ROOT_DIR}" "${REPO_DIR}/slurm_runs/lyris"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 8)); then
    echo "Expected eight allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
RUN_NODELIST=$(IFS=,; echo "${allocated_nodes[*]}")
FIRST_NODE="${allocated_nodes[0]}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

# Warm all node-local Enroot caches once and run read-only implementation and
# placement preflights before consuming the allocation on either timed case.
srun --nodes=8 --nodelist="${RUN_NODELIST}" --ntasks=8 --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc 'python -c "import torch, mamba_ssm, causal_conv1d; print(torch.__version__)"'

srun --nodes=1 --nodelist="${FIRST_NODE}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${REPO_DIR}'
python -m py_compile \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
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
assert [len(row) for row in full_slots] == [8] * 16
assert [len(row) for row in reduced_slots] == [11] * 12
assert sum(expert is None for row in reduced_slots for expert in row) == 4
print(\"[a3b-distopt-ab] implementation and placement preflight: PASS\")
'
"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT="${NCCL_LAUNCH_ORDER_IMPLICIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0

# Preserve the performance-retaining NEP settings validated by the DistOpt
# smoke.  Healthy receives the same environment; NEP-only settings are inert.
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0
export MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0
export MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1
export MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1
export MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0
export MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0
export MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1
export MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0
export MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1
export MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW=1
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=8
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter

run_case() {
    local label="$1"
    local run_nodes="$2"
    local world_size="$3"
    local topology="$4"
    local nonuniform_mode="$5"
    local global_batch_size="$6"
    local master_port="$7"
    local profile_ranks
    local name

    profile_ranks=$(seq -s ' ' 0 $((world_size - 1)))
    name="a3b_30b_128e_${label}_distopt_i${TRAIN_ITERS}"

    echo "[a3b-distopt-ab] $(date --iso-8601=seconds) starting ${label}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}/${label}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${name}" \
            MASTER_ADDR="${FIRST_NODE}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${run_nodes}" \
            RUN_WORLD_SIZE="${world_size}" \
            GPUS_PER_NODE=4 \
            NVLINK_SEGMENT_NODES=4 \
            USE_DIRECT_SRUN_RANKS=1 \
            TRAIN_ITERS="${TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS=2 \
            HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN}" \
            NUM_EXPERTS=128 \
            MOE_ROUTER_TOPK=6 \
            SEQ_LENGTH=8192 \
            NONUNIFORM_MODE="${nonuniform_mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=16 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE=1 \
            GLOBAL_BATCH_SIZE="${global_batch_size}" \
            TRUE_GLOBAL_BATCH_SIZE="${global_batch_size}" \
            REPLICA_MICRO_BATCH_SIZES="1 1" \
            REPLICA_NUM_MICROBATCHES="1 1" \
            DDP_NUM_BUCKETS=8 \
            CUDA_GRAPH_IMPL=none \
            PROFILE=1 \
            PROFILE_STEP_START="${PROFILE_STEP_START}" \
            PROFILE_STEP_END="${PROFILE_STEP_END}" \
            PROFILE_RANKS="${profile_ranks}" \
            LOG_PARAMS_NORM=0 \
            LOG_NUM_ZEROS_IN_GRAD=0 \
            LOG_ENERGY=0 \
            MANUAL_GC_INTERVAL=1000 \
            LOG_INTERVAL=1 \
            EXTRA_MEGATRON_ARGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --calculate-per-token-loss --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-transformer-engine-op-fuser --moe-expert-rank-capacity-factor ${MOE_EXPERT_RANK_CAPACITY_FACTOR} --use-distributed-optimizer --overlap-param-gather --no-check-for-nan-in-loss-and-grad --rerun-mode disabled --nonuniform-disable-nongrad-sync-collectives" \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
            MOE_ROUTER_FORCE_UNIFORM_ROUTING=0 \
            MOE_ROUTER_ENABLE_EXPERT_BIAS=0 \
            NONUNIFORM_SKIP_OPTIMIZER_STEP=0 \
            MEGATRON_NONUNIFORM_EP_DEBUG=0 \
            MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="${NEP_DEBUG_RANKS}" \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS=16 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT=1 \
            bash "${RUNNER}"
    echo "[a3b-distopt-ab] $(date --iso-8601=seconds) completed ${label}"
}

if [[ "${RUN_HEALTHY_CASE}" == "1" ]]; then
    run_case healthy_ep16_ep16 8 32 "8 8" none 16 31161
fi
if [[ "${RUN_NEP_CASE}" == "1" ]]; then
    run_case nep_ep16_ep12 7 28 "8 6" ep 14 31162
fi

if [[ "${RUN_HEALTHY_CASE}" != "1" || "${RUN_NEP_CASE}" != "1" ]]; then
    echo "[a3b-distopt-ab] selected cases completed; skipping paired comparison"
    exit 0
fi

HEALTHY_NAME="a3b_30b_128e_healthy_ep16_ep16_distopt_i${TRAIN_ITERS}"
NEP_NAME="a3b_30b_128e_nep_ep16_ep12_distopt_i${TRAIN_ITERS}"
HEALTHY_RUN="${ROOT_DIR}/healthy_ep16_ep16/${HEALTHY_NAME}/${SLURM_JOB_ID}"
NEP_RUN="${ROOT_DIR}/nep_ep16_ep12/${NEP_NAME}/${SLURM_JOB_ID}"
RESULT_JSON="${ROOT_DIR}/comparison_${SLURM_JOB_ID}.json"

python3 - "${HEALTHY_RUN}" "${NEP_RUN}" "${TRAIN_ITERS}" "${RESULT_JSON}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

healthy_run = Path(sys.argv[1])
nep_run = Path(sys.argv[2])
expected_iters = int(sys.argv[3])
output = Path(sys.argv[4])
timing_start = 8
timing_pattern = re.compile(
    r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
iteration_status = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)

result = {}
for label, run, world_size, mode in (
    ("healthy", healthy_run, 32, "none"),
    ("nep", nep_run, 28, "ep"),
):
    # RUN_DIR ends in the Slurm job id, which is also embedded in the driver filename.
    log = run / f"driver_{run.name}.log"
    if not log.is_file():
        candidates = sorted(run.glob("driver_*.log"))
        if len(candidates) != 1:
            raise RuntimeError(f"{label}: expected one driver log in {run}, got {candidates}")
        log = candidates[0]
    text = log.read_text()
    required = {
        "DistOpt": r"use_distributed_optimizer\s+\.+\s+True",
        "overlap grad reduce": r"overlap_grad_reduce\s+\.+\s+True",
        "overlap param gather": r"overlap_param_gather\s+\.+\s+True",
        "Flex": r"moe_token_dispatcher_type\s+\.+\s+flex",
        "HybridEP": r"moe_flex_dispatcher_backend\s+\.+\s+hybridep",
        "TE op fuser": r"use_transformer_engine_op_fuser\s+\.+\s+True",
        "static rank capacity": r"moe_expert_rank_capacity_factor\s+\.+\s+1\.5",
        "disabled overflow checking": r"check_for_nan_in_loss_and_grad\s+\.+\s+False",
        "mode": rf"nonuniform_mode\s+\.+\s+{mode}",
    }
    for description, regex in required.items():
        if re.search(regex, text) is None:
            raise RuntimeError(f"{label}: missing runtime confirmation for {description}")
    if "token drop during MoE token dispatch" in text or "rerunning forward-backward" in text:
        raise RuntimeError(f"{label}: static HybridEP capacity overflowed")
    statuses = [tuple(map(int, match.groups())) for match in iteration_status.finditer(text)]
    if [row[0] for row in statuses[-expected_iters:]] != list(range(1, expected_iters + 1)):
        raise RuntimeError(f"{label}: missing iterations: {statuses}")
    if any(row[1] != expected_iters or row[2] or row[3] for row in statuses[-expected_iters:]):
        raise RuntimeError(f"{label}: invalid iteration status: {statuses[-expected_iters:]}")
    rows = [
        {"iteration": int(i), "elapsed_ms": float(ms)}
        for i, ms in timing_pattern.findall(text)
        if int(i) >= timing_start
    ]
    expected_timing = list(range(timing_start, expected_iters + 1))
    if [row["iteration"] for row in rows] != expected_timing:
        raise RuntimeError(f"{label}: expected timing iterations {expected_timing}, got {rows}")
    traces = sorted((run / "torch_profile").glob("rank-*.json.gz"))
    if len(traces) != world_size:
        raise RuntimeError(f"{label}: expected {world_size} traces, got {len(traces)}")
    result[label] = {
        "run_dir": str(run),
        "rows": rows,
        "mean_elapsed_ms": statistics.mean(row["elapsed_ms"] for row in rows),
        "std_elapsed_ms": statistics.stdev(row["elapsed_ms"] for row in rows),
        "trace_count": len(traces),
    }

healthy_ms = result["healthy"]["mean_elapsed_ms"]
nep_ms = result["nep"]["mean_elapsed_ms"]
result["comparison"] = {
    "latency_gap_ms": nep_ms - healthy_ms,
    "nep_slowdown_percent": 100.0 * (nep_ms / healthy_ms - 1.0),
    "owner_iteration_time_parity_percent": 100.0 * healthy_ms / nep_ms,
    "relative_throughput_percent": 100.0 * healthy_ms / nep_ms,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result["comparison"], sort_keys=True))
print(f"[a3b-distopt-ab] PASS: wrote {output}")
PY

echo "[a3b-distopt-ab] complete: ${RESULT_JSON}"
