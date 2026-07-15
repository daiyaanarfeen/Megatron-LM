#!/bin/bash

# Lyris GB200 wrapper for the profiled EP8/4 NEP overlap diagnostic.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=lyris_nep_overlap_smoke

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_nep_smoke}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:25.09}"
CONTAINER_NAME="${CONTAINER_NAME:-}"
NAME="${NAME:-ep8_4_l16_h1024_s1024_nonblocking_slots_${SLURM_JOB_ID}}"
RUN_DIR="${ROOT_DIR}/${NAME}"
DRIVER_LOG="${RUN_DIR}/driver_${SLURM_JOB_ID}.log"
MASTER_PORT="${MASTER_PORT:-29890}"
RUN_NNODES="${RUN_NNODES:-${SLURM_NNODES}}"
RUN_NPROC_PER_NODE="${RUN_NPROC_PER_NODE:-4}"
RUN_WORLD_SIZE="${RUN_WORLD_SIZE:-$((RUN_NNODES * RUN_NPROC_PER_NODE))}"
USE_DIRECT_SRUN_RANKS="${USE_DIRECT_SRUN_RANKS:-0}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-8 4}"
FORMAT_SOURCES="${FORMAT_SOURCES:-0}"
RUN_PREFLIGHT_TESTS="${RUN_PREFLIGHT_TESTS:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-1}"
EXTRA_MEGATRON_ARGS="${EXTRA_MEGATRON_ARGS:-}"
export FORMAT_SOURCES

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((RUN_NNODES > ${#allocated_nodes[@]})); then
    echo "RUN_NNODES=${RUN_NNODES} exceeds the ${#allocated_nodes[@]} allocated nodes" >&2
    exit 2
fi
run_nodes=("${allocated_nodes[@]:0:RUN_NNODES}")
RUN_NODELIST=$(IFS=,; echo "${run_nodes[*]}")
MASTER_ADDR="${MASTER_ADDR:-${run_nodes[0]}}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

finish() {
    status=$?
    trap - EXIT
    echo "[lyris-nep-smoke] exit status: ${status}"
    exit "${status}"
}
trap finish EXIT

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

echo "[lyris-nep-smoke] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} image=${IMAGE}"

if [[ "${RUN_PREFLIGHT_TESTS}" == "1" ]]; then
    srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" bash -lc '
set -euo pipefail
cd /home/darfeen/Megatron-LM
if [[ "${FORMAT_SOURCES}" == "1" ]]; then
    uv run isort \
        megatron/core/transformer/moe/moe_layer.py \
        megatron/core/transformer/moe/token_dispatcher.py \
        megatron/core/distributed/_cuda_stream_ops.py \
        megatron/core/distributed/nonuniform_common.py \
        megatron/core/distributed/nonuniform_ep.py \
        scripts/nonuniform/probe_cuda_stream_ops.py \
        tests/unit_tests/distributed/test_nonuniform_ep.py
    python -m black \
        megatron/core/transformer/moe/moe_layer.py \
        megatron/core/transformer/moe/token_dispatcher.py \
        megatron/core/distributed/_cuda_stream_ops.py \
        megatron/core/distributed/nonuniform_common.py \
        megatron/core/distributed/nonuniform_ep.py \
        scripts/nonuniform/probe_cuda_stream_ops.py \
        tests/unit_tests/distributed/test_nonuniform_ep.py
fi
python -m black --check \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    megatron/core/distributed/_cuda_stream_ops.py \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    scripts/nonuniform/probe_cuda_stream_ops.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m compileall -q \
    megatron/core/distributed/_native_nccl.py \
    megatron/core/distributed/_cuda_stream_ops.py \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    scripts/nonuniform/probe_cuda_stream_ops.py
python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py
python - <<'"'"'PY'"'"'
from megatron.core.distributed.nonuniform_ep import NonuniformEPNCCLParamAndGradBucketGroup


class Work:
    def __init__(self):
        self.block_calls = 0
        self.wait_calls = 0

    def block_current_stream(self):
        self.block_calls += 1

    def wait(self):
        self.wait_calls += 1


group = NonuniformEPNCCLParamAndGradBucketGroup.__new__(
    NonuniformEPNCCLParamAndGradBucketGroup
)
key = (0, 128, None, None)
works = [Work(), Work()]
group._nep_nccl_scheduler_state = {
    "gather_buf_cache": {},
    "buffer_slot_handles": {key: works},
}
group._order_nep_nccl_buffer_slot(key)
assert all(work.block_calls == 1 and work.wait_calls == 0 for work in works)
print("[lyris-nep-smoke] nonblocking slot test: ok")
PY
'
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    exit 0
fi

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR=/tmp/triton_cache
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW="${MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW:-16}"
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG="${MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG:-1}"

export RUN_DIRECT=1
export IMAGE_PATH="${IMAGE}"
export ROOT_DIR REPO_DIR NAME MASTER_ADDR MASTER_PORT IMAGE_PATH
export NNODES="${RUN_NNODES}"
export GPUS_PER_NODE=4
export NPROC_PER_NODE="${RUN_NPROC_PER_NODE}"
export TRAIN_ITERS="${TRAIN_ITERS:-10}"
export GLOBAL_BATCH_SIZE
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export NUM_LAYERS="${NUM_LAYERS:-16}"
export HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-4096}"
export NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
export TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
export EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
export SEQ_LENGTH="${SEQ_LENGTH:-1024}"
export NUM_EXPERTS NONUNIFORM_EP_TOPOLOGY
export ENABLE_PYTORCH_PROFILER
export PROFILE_STEP_START="${PROFILE_STEP_START:-4}"
export PROFILE_STEP_END="${PROFILE_STEP_END:-6}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export EXTRA_MEGATRON_ARGS="--nonuniform-skip-optimizer-step ${EXTRA_MEGATRON_ARGS}"

if [[ "${USE_DIRECT_SRUN_RANKS}" == "1" ]]; then
    export LAUNCHER_MODE=direct
    export WORLD_SIZE="${RUN_WORLD_SIZE}"
    srun --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
        --ntasks="${RUN_WORLD_SIZE}" --ntasks-per-node="${RUN_NPROC_PER_NODE}" \
        --kill-on-bad-exit=1 \
        --mpi=none "${container_args[@]}" \
        bash -lc 'export RANK="${SLURM_PROCID}" LOCAL_RANK=0 TORCH_CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"; bash examples/training_scripts/nonuniform_ep_approach_a_smoke.sh'
else
    srun --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
        --ntasks="${RUN_NNODES}" --ntasks-per-node=1 \
        --kill-on-bad-exit=1 --mpi=none "${container_args[@]}" \
        bash examples/training_scripts/nonuniform_ep_approach_a_smoke.sh
fi
