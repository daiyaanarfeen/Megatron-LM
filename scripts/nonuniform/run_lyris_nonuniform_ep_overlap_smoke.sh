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
NAME="${NAME:-ep8_4_l16_h1024_s1024_nonblocking_slots_${SLURM_JOB_ID}}"
RUN_DIR="${ROOT_DIR}/${NAME}"
DRIVER_LOG="${RUN_DIR}/driver_${SLURM_JOB_ID}.log"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
MASTER_PORT="${MASTER_PORT:-29890}"
RUN_NNODES="${RUN_NNODES:-${SLURM_NNODES}}"
RUN_NPROC_PER_NODE="${RUN_NPROC_PER_NODE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-8 4}"

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

echo "[lyris-nep-smoke] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST} image=${IMAGE}"

srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" bash -lc '
cd /home/darfeen/Megatron-LM
python -m compileall -q megatron/core/distributed/nonuniform_ep.py
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

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_WORKER_START=fork
export TRITON_CACHE_DIR=/tmp/triton_cache
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=1

export RUN_DIRECT=1
export IMAGE_PATH="${IMAGE}"
export ROOT_DIR REPO_DIR NAME MASTER_ADDR MASTER_PORT IMAGE_PATH
export NNODES="${RUN_NNODES}"
export GPUS_PER_NODE=4
export NPROC_PER_NODE="${RUN_NPROC_PER_NODE}"
export TRAIN_ITERS=10
export GLOBAL_BATCH_SIZE
export MICRO_BATCH_SIZE=1
export NUM_LAYERS=16
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=16
export SEQ_LENGTH=1024
export NUM_EXPERTS NONUNIFORM_EP_TOPOLOGY
export ENABLE_PYTORCH_PROFILER=1
export PROFILE_STEP_START=4
export PROFILE_STEP_END=6
export PROFILE_RANKS=0
export EXTRA_MEGATRON_ARGS="--nonuniform-skip-optimizer-step"

srun --nodes="${RUN_NNODES}" --ntasks="${RUN_NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --mpi=none "${container_args[@]}" bash examples/training_scripts/nonuniform_ep_approach_a_smoke.sh
