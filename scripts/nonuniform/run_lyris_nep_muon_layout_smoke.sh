#!/bin/bash

# Small EP8/EP6 integration gate for NEP with Flex and LayerWise DistOpt (Muon).

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err

set -euo pipefail

BENCH_REPO="${BENCH_REPO:-/home/darfeen/Megatron-LM}"
CODE_REPO="${CODE_REPO:-/home/darfeen/Megatron-LM-nep-layerwise-20260826}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/lustre/fsw/coreai_comparch_sysarch/darfeen}"
ROOT_DIR="${ROOT_DIR:-${ARTIFACT_ROOT}/slurm_runs/nep_muon_layout_smoke}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-8m}"
FULL_EP_SIZE="${FULL_EP_SIZE:-8}"
REDUCED_EP_SIZE="${REDUCED_EP_SIZE:-6}"
WORLD_SIZE=$((FULL_EP_SIZE + REDUCED_EP_SIZE))
NODE_COUNT="${NODE_COUNT:-4}"
TASKS_PER_NODE="${TASKS_PER_NODE:-4}"
ACTIVE_NODE_COUNT=$(((WORLD_SIZE + TASKS_PER_NODE - 1) / TASKS_PER_NODE))
NUM_EXPERTS="${NUM_EXPERTS:-$((2 * FULL_EP_SIZE))}"
PROFILE_RANKS=$(seq -s ' ' 0 $((WORLD_SIZE - 1)))
RUN_DIR="${ROOT_DIR}/${SLURM_JOB_ID}"
DRIVER_LOG="${RUN_DIR}/driver.log"

mkdir -p "${RUN_DIR}/tensorboard" "${RUN_DIR}/torch_profile"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${BENCH_REPO}:${BENCH_REPO},${CODE_REPO}:${CODE_REPO},${ARTIFACT_ROOT}:${ARTIFACT_ROOT}"
    --container-workdir="${CODE_REPO}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#nodes[@]} != NODE_COUNT)); then
    echo "Expected ${NODE_COUNT} allocated nodes, got ${#nodes[@]}" >&2
    exit 2
fi
active_nodes=("${nodes[@]:0:ACTIVE_NODE_COUNT}")
nodelist=$(IFS=,; echo "${active_nodes[*]}")
master_addr="${active_nodes[0]}"

srun --nodes="${ACTIVE_NODE_COUNT}" --nodelist="${nodelist}" --ntasks="${ACTIVE_NODE_COUNT}" --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc 'python -c "import torch, transformer_engine; print(torch.__version__)"'

srun --nodes=1 --nodelist="${master_addr}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${CODE_REPO}'
python -m py_compile \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/layer_wise_optimizer.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py
"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export USE_MNNVL=1

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
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=2
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter

options="--use-mcore-models \
--num-layers 2 \
--hidden-size 512 \
--ffn-hidden-size 1024 \
--num-attention-heads 8 \
--seq-length 128 \
--max-position-embeddings 128 \
--position-embedding-type rope \
--normalization RMSNorm \
--disable-bias-linear \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--transformer-impl transformer_engine \
--attention-backend unfused \
--bf16 \
--num-experts ${NUM_EXPERTS} \
--moe-router-topk 2 \
--moe-router-pre-softmax \
--moe-router-force-load-balancing \
--moe-router-load-balancing-type aux_loss \
--moe-aux-loss-coeff 0.01 \
--moe-token-dispatcher-type flex \
--moe-flex-dispatcher-backend hybridep \
--moe-router-dtype fp32 \
--moe-grouped-gemm \
--moe-permute-fusion \
--optimizer muon \
--muon-momentum 0.9 \
--muon-extra-scale-factor 0.2 \
--muon-scale-mode spectral \
--muon-num-ns-steps 2 \
--muon-scalar-optimizer adam \
--use-distributed-optimizer \
--overlap-grad-reduce \
--overlap-param-gather \
--ddp-num-buckets 2 \
--tensor-model-parallel-size 1 \
--context-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--expert-model-parallel-size ${FULL_EP_SIZE} \
--expert-tensor-parallel-size 1 \
--high-priority-stream-groups ep \
--nonuniform-mode ep \
--nonuniform-ep-ddp-approach nccl \
--nonuniform-ep-num-tp-cp-per-replica ${FULL_EP_SIZE} ${REDUCED_EP_SIZE} \
--calculate-per-token-loss \
--mock-data \
--tokenizer-type NullTokenizer \
--vocab-size 4096 \
--num-workers 1 \
--train-iters 4 \
--lr 1.0e-4 \
--min-lr 1.0e-5 \
--lr-decay-style constant \
--eval-interval 1000 \
--eval-iters 0 \
--log-interval 1 \
--timing-log-option minmax \
--tensorboard-dir ${RUN_DIR}/tensorboard \
--profile \
--use-pytorch-profiler \
--profile-step-start 1 \
--profile-step-end 3 \
--profile-ranks ${PROFILE_RANKS} \
--no-check-for-nan-in-loss-and-grad \
--manual-gc \
--manual-gc-interval 1000 \
--distributed-timeout-minutes 3 \
--rerun-mode disabled"

export REPO_DIR="${CODE_REPO}"
export PRETRAIN_ENTRYPOINT="${CODE_REPO}/examples/nonuniform/pretrain_gpt_nonuniform.py"
export RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_lyris_training_scripts_nep_rank.sh"
export BASE_RANK_LAUNCHER="${BENCH_REPO}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh"
export options
export NONUNIFORM_MODE=ep
export NONUNIFORM_EP_TOPOLOGY="${FULL_EP_SIZE} ${REDUCED_EP_SIZE}"
export TENSOR_MODEL_PARALLEL_SIZE=1
export CONTEXT_PARALLEL_SIZE=1
export EXPERT_TENSOR_PARALLEL_SIZE=1
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE="${WORLD_SIZE}"
export TRUE_GLOBAL_BATCH_SIZE="${WORLD_SIZE}"
export REPLICA_MICRO_BATCH_SIZES="1 1"
export REPLICA_NUM_MICROBATCHES="1 1"
export MASTER_ADDR="${master_addr}"
export MASTER_PORT=32470

set +e
# The SLURM rank variables expand inside each task, not in this driver shell.
# shellcheck disable=SC2016
timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
    srun --overlap --nodes="${ACTIVE_NODE_COUNT}" --nodelist="${nodelist}" \
    --ntasks="${WORLD_SIZE}" --ntasks-per-node="${TASKS_PER_NODE}" --kill-on-bad-exit=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc 'export RANK="${SLURM_PROCID}" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"; exec bash "${RANK_LAUNCHER}"' \
    2>&1 | tee "${DRIVER_LOG}"
status=${PIPESTATUS[0]}
set -e
if ((status != 0)); then
    echo "Muon NEP smoke failed with status ${status}" >&2
    exit "${status}"
fi

python3 - "${DRIVER_LOG}" "${RUN_DIR}" "${FULL_EP_SIZE}" "${REDUCED_EP_SIZE}" "${WORLD_SIZE}" <<'PY'
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
full_ep_size = int(sys.argv[3])
reduced_ep_size = int(sys.argv[4])
world_size = int(sys.argv[5])
text = log_path.read_text(errors="replace")

required = {
    "LayerWise optimizer": r"use_layer_wise_distributed_optimizer\s+\.+\s+True",
    "LayerWise layout": r"use_layer_wise_param_layout\s+\.+\s+True",
    "Flex dispatcher": r"moe_token_dispatcher_type\s+\.+\s+flex",
    "HybridEP backend": r"moe_flex_dispatcher_backend\s+\.+\s+hybridep",
}
for name, pattern in required.items():
    if re.search(pattern, text) is None:
        raise RuntimeError(f"runtime did not confirm {name}")
if "crosses shard boundary" in text:
    raise RuntimeError("LayerWise parameter layout still crosses an optimizer shard")

status_pattern = re.compile(
    r"iteration\s+(\d+)/\s*4.*?number of skipped iterations:\s*(\d+).*?"
    r"number of nan iterations:\s*(\d+)"
)
statuses = [tuple(map(int, match.groups())) for match in status_pattern.finditer(text)][-4:]
if statuses != [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]:
    raise RuntimeError(f"incomplete or invalid training iterations: {statuses}")

traces = (
    list(run_dir.rglob("rank-*.json.gz"))
    + list(run_dir.rglob("*.pt.trace.json"))
    + list(run_dir.rglob("*.pt.trace.json.gz"))
)
if len(set(traces)) != world_size:
    raise RuntimeError(
        f"expected {world_size} all-rank traces, found {len(set(traces))}"
    )

timings = {
    int(iteration): float(elapsed)
    for iteration, elapsed in re.findall(
        r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\):\s*([0-9.]+)", text
    )
}
result = {
    "job_id": run_dir.name,
    "topology": [full_ep_size, reduced_ep_size],
    "iterations_ms": timings,
    "trace_count": len(set(traces)),
    "validated_iterations": [status[0] for status in statuses],
}
(run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
PY

echo "PASS: EP${FULL_EP_SIZE}/EP${REDUCED_EP_SIZE} Flex LayerWise-DistOpt NEP smoke"
