#!/bin/bash

# dlcluster wrapper for the nonuniform EP overlap diagnostic.
#
# This stages the current repo and container image to node-local /tmp, then runs
# examples/training_scripts/nonuniform_ep_approach_a_smoke.sh in RUN_DIRECT mode
# with PyTorch profiler enabled.  The workload is deliberately larger than the
# correctness smoke so there is later backward compute that can hide NEP reshard
# communication if the async implementation is actually overlapping.

#SBATCH --account=blackwell
#SBATCH --partition=gb200nvl72
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/%x-%j.out
#SBATCH --error=/tmp/%x-%j.err
#SBATCH --job-name=dl_nep_overlap_diag

set -euo pipefail

LOGIN_HOST="${LOGIN_HOST:-dlcluster-login-01}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/tmp/darfeen_known_hosts}"
# Reuse the same node-local root as the validated dlcluster smoke wrapper so
# nodes with an existing named enroot container can skip the 29.6 GB image copy.
STAGE_ROOT="${STAGE_ROOT:-/tmp/darfeen_mep_hsg_a8b}"
REPO_ARCHIVE_SRC="${REPO_ARCHIVE_SRC:-${LOGIN_HOST}:/tmp/megatron_ep_stage.tar.gz}"
IMAGE_SRC="${IMAGE_SRC:-${LOGIN_HOST}:/home/scratch.darfeen_gpu/images/nvidia+pytorch+25.06-py3+dependencies+mamba.sqsh}"
IMAGE_BYTES="${IMAGE_BYTES:-29622525952}"
RESULTS_BASE_PATH="${RESULTS_BASE_PATH:-/home/scratch.darfeen_gpu/dlcluster_runs}"

CONTAINER_NAME="${CONTAINER_NAME:-nvidia-pytorch-25-06-deps-mamba}"
IMAGE_PATH="${STAGE_ROOT}/image.sqsh"
REPO_DIR="${STAGE_ROOT}/repo"
ROOT_DIR="${STAGE_ROOT}/runs"
RUN_NNODES="${RUN_NNODES:-2}"
RUN_GPUS_PER_NODE="${RUN_GPUS_PER_NODE:-4}"
RUN_NPROC_PER_NODE="${RUN_NPROC_PER_NODE:-3}"
NAME="${NAME:-dlcluster_nep_overlap_diag_ep4_ep2_l16_h1024_s1024}"
MASTER_PORT="${MASTER_PORT:-29690}"
DRIVER_LOG="${STAGE_ROOT}/driver_${SLURM_JOB_ID}.log"
USE_DIRECT_SRUN_RANKS="${USE_DIRECT_SRUN_RANKS:-1}"
TRAIN_ITERS="${TRAIN_ITERS:-20}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-6}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
NUM_LAYERS="${NUM_LAYERS:-16}"
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-4096}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
NONUNIFORM_EP_TOPOLOGY="${NONUNIFORM_EP_TOPOLOGY:-4 2}"
ENABLE_PYTORCH_PROFILER="${ENABLE_PYTORCH_PROFILER:-1}"
ENABLE_NSYS_PROFILE="${ENABLE_NSYS_PROFILE:-0}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-none}"
PROFILE_STEP_START="${PROFILE_STEP_START:-10}"
PROFILE_STEP_END="${PROFILE_STEP_END:-12}"
PROFILE_RANKS="${PROFILE_RANKS:-0 1 2 3 4 5}"
EXTRA_SMOKE_ARGS="${EXTRA_SMOKE_ARGS:-}"

mkdir -p "${STAGE_ROOT}"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

copy_logs_back() {
    status=$?
    set +e
    echo "[dlcluster-nep-overlap] exit status: ${status}"
    echo "[dlcluster-nep-overlap] copying run dir back to ${LOGIN_HOST}:${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}/"
    ssh ${SSH_OPTS} "${LOGIN_HOST}" "mkdir -p '${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}'"

    gather_node_logs='
set +e
node="$(hostname)"
archive="'"${STAGE_ROOT}"'/logs_'"${SLURM_JOB_ID}"'_${node}.tgz"
cd "'"${STAGE_ROOT}"'" || exit 0
paths=()
if [[ -f "'"$(basename "${DRIVER_LOG}")"'" ]]; then
    paths+=("'"$(basename "${DRIVER_LOG}")"'")
fi
if [[ -d "runs/'"${NAME}"'" ]]; then
    paths+=("runs/'"${NAME}"'")
fi
if (( ${#paths[@]} > 0 )); then
    tar -czf "${archive}" "${paths[@]}" 2>/dev/null
    rsync -av -e "ssh '"${SSH_OPTS}"'" "${archive}" "'"${LOGIN_HOST}"':'"${RESULTS_BASE_PATH}"'/'"${NAME}"'/'"${SLURM_JOB_ID}"'/"
fi
'
    srun \
        --overlap \
        --nodes="${SLURM_NNODES:-1}" \
        --ntasks="${SLURM_NNODES:-1}" \
        --ntasks-per-node=1 \
        --mpi=none \
        bash -lc "${gather_node_logs}" || true

    tar -czf "${STAGE_ROOT}/logs_${SLURM_JOB_ID}.tgz" -C "${STAGE_ROOT}" \
        "$(basename "${DRIVER_LOG}")" \
        "runs/${NAME}" 2>/dev/null
    rsync -av -e "ssh ${SSH_OPTS}" "${STAGE_ROOT}/logs_${SLURM_JOB_ID}.tgz" \
        "${LOGIN_HOST}:${RESULTS_BASE_PATH}/${NAME}/${SLURM_JOB_ID}/"
    exit "${status}"
}
trap copy_logs_back EXIT

echo "[dlcluster-nep-overlap] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[dlcluster-nep-overlap] stage_root=${STAGE_ROOT}"

stage_one_node='
set -euo pipefail
mkdir -p "'"${STAGE_ROOT}"'" "'"${ROOT_DIR}"'"
export ENROOT_CACHE_PATH="'"${STAGE_ROOT}"'/enroot-cache"
export ENROOT_DATA_PATH="'"${STAGE_ROOT}"'/enroot-data"
export ENROOT_RUNTIME_PATH="'"${STAGE_ROOT}"'/enroot-runtime"
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"
container_exists=0
if enroot list | grep -qx "'"${CONTAINER_NAME}"'"; then
    container_exists=1
fi
if [[ "${container_exists}" != "1" ]] && { [[ ! -f "'"${IMAGE_PATH}"'" ]] || [[ "$(stat -c%s "'"${IMAGE_PATH}"'" 2>/dev/null || echo 0)" != "'"${IMAGE_BYTES}"'" ]]; }; then
    echo "[stage $(hostname)] syncing image"
    rsync -av --partial --append-verify --info=progress2 -e "ssh '"${SSH_OPTS}"'" "'"${IMAGE_SRC}"'" "'"${IMAGE_PATH}"'"
elif [[ "${container_exists}" == "1" ]]; then
    echo "[stage $(hostname)] enroot container exists; skipping image sync"
fi
echo "[stage $(hostname)] syncing repo archive"
rsync -av -e "ssh '"${SSH_OPTS}"'" "'"${REPO_ARCHIVE_SRC}"'" "'"${STAGE_ROOT}"'/repo.tar.gz"
rm -rf "'"${REPO_DIR}"'"
mkdir -p "'"${REPO_DIR}"'"
tar -xzf "'"${STAGE_ROOT}"'/repo.tar.gz" -C "'"${REPO_DIR}"'" --strip-components=1
if ! enroot list | grep -qx "'"${CONTAINER_NAME}"'"; then
    echo "[stage $(hostname)] creating enroot container "'"${CONTAINER_NAME}"'""
    enroot create --name "'"${CONTAINER_NAME}"'" "'"${IMAGE_PATH}"'"
else
    echo "[stage $(hostname)] enroot container already exists"
fi
if ! compgen -G "'"${REPO_DIR}"'/megatron/core/datasets/helpers_cpp*.so" >/dev/null; then
    echo "[stage $(hostname)] compiling dataset helper extension"
    enroot start --rw --mount "'"${STAGE_ROOT}"':'"${STAGE_ROOT}"'" "'"${CONTAINER_NAME}"'" \
        sh -lc "cd '"${REPO_DIR}"' && python -c \"from megatron.core.datasets.utils import compile_helpers; compile_helpers()\""
else
    echo "[stage $(hostname)] dataset helper extension already exists"
fi
'

echo "[dlcluster-nep-overlap] staging all nodes"
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --mpi=none \
    bash -lc "${stage_one_node}"

MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)}"
run_cmd="cd ${REPO_DIR} && \
RUN_DIRECT=1 \
ROOT_DIR=${ROOT_DIR} \
REPO_DIR=${REPO_DIR} \
IMAGE_PATH=${IMAGE_PATH} \
CONTAINER_NAME= \
CONTAINER_MOUNTS=${STAGE_ROOT}:${STAGE_ROOT} \
NNODES=${RUN_NNODES} \
GPUS_PER_NODE=${RUN_GPUS_PER_NODE} \
NPROC_PER_NODE=${RUN_NPROC_PER_NODE} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
CUDA_DEVICE_MAX_CONNECTIONS=32 \
MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=1 \
TRAIN_ITERS=${TRAIN_ITERS} \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} \
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} \
NUM_LAYERS=${NUM_LAYERS} \
HIDDEN_SIZE=${HIDDEN_SIZE} \
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE} \
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS} \
SEQ_LENGTH=${SEQ_LENGTH} \
NUM_EXPERTS=${NUM_EXPERTS} \
NONUNIFORM_EP_TOPOLOGY='${NONUNIFORM_EP_TOPOLOGY}' \
ENABLE_PYTORCH_PROFILER=${ENABLE_PYTORCH_PROFILER} \
ENABLE_NSYS_PROFILE=${ENABLE_NSYS_PROFILE} \
NSYS_CAPTURE_RANGE=${NSYS_CAPTURE_RANGE} \
PROFILE_STEP_START=${PROFILE_STEP_START} \
PROFILE_STEP_END=${PROFILE_STEP_END} \
PROFILE_RANKS='${PROFILE_RANKS}' \
EXTRA_MEGATRON_ARGS='${EXTRA_SMOKE_ARGS}' \
NAME=${NAME} \
bash examples/training_scripts/nonuniform_ep_approach_a_smoke.sh"

echo "[dlcluster-nep-overlap] launching diagnostic: nodes=${RUN_NNODES} nproc_per_node=${RUN_NPROC_PER_NODE} direct_srun_ranks=${USE_DIRECT_SRUN_RANKS}"
export ENROOT_CACHE_PATH="${STAGE_ROOT}/enroot-cache"
export ENROOT_DATA_PATH="${STAGE_ROOT}/enroot-data"
export ENROOT_RUNTIME_PATH="${STAGE_ROOT}/enroot-runtime"
if [[ "${USE_DIRECT_SRUN_RANKS}" == "1" ]]; then
    WORLD_SIZE="$((RUN_NNODES * RUN_NPROC_PER_NODE))"
    direct_run_cmd="cd ${REPO_DIR} && \
RUN_DIRECT=1 \
LAUNCHER_MODE=direct \
ROOT_DIR=${ROOT_DIR} \
REPO_DIR=${REPO_DIR} \
IMAGE_PATH=${IMAGE_PATH} \
CONTAINER_NAME= \
CONTAINER_MOUNTS=${STAGE_ROOT}:${STAGE_ROOT} \
NNODES=${RUN_NNODES} \
GPUS_PER_NODE=${RUN_NPROC_PER_NODE} \
NPROC_PER_NODE=${RUN_NPROC_PER_NODE} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
WORLD_SIZE=${WORLD_SIZE} \
RANK=\${SLURM_PROCID} \
LOCAL_RANK=0 \
TORCH_CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID} \
CUDA_DEVICE_MAX_CONNECTIONS=32 \
MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=1 \
TRAIN_ITERS=${TRAIN_ITERS} \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} \
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} \
NUM_LAYERS=${NUM_LAYERS} \
HIDDEN_SIZE=${HIDDEN_SIZE} \
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE} \
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS} \
SEQ_LENGTH=${SEQ_LENGTH} \
NUM_EXPERTS=${NUM_EXPERTS} \
NONUNIFORM_EP_TOPOLOGY='${NONUNIFORM_EP_TOPOLOGY}' \
ENABLE_PYTORCH_PROFILER=${ENABLE_PYTORCH_PROFILER} \
ENABLE_NSYS_PROFILE=${ENABLE_NSYS_PROFILE} \
NSYS_CAPTURE_RANGE=${NSYS_CAPTURE_RANGE} \
NSYS_OUTPUT_NAME=rank_\${SLURM_PROCID} \
PROFILE_STEP_START=${PROFILE_STEP_START} \
PROFILE_STEP_END=${PROFILE_STEP_END} \
PROFILE_RANKS='${PROFILE_RANKS}' \
EXTRA_MEGATRON_ARGS='${EXTRA_SMOKE_ARGS}' \
NAME=${NAME} \
bash examples/training_scripts/nonuniform_ep_approach_a_smoke.sh"
    srun -l \
        --nodes="${RUN_NNODES}" \
        --ntasks="${WORLD_SIZE}" \
        --ntasks-per-node="${RUN_NPROC_PER_NODE}" \
        --gpus-per-task=1 \
        --gpu-bind=single:1 \
        --mpi=none \
        --container-image "${IMAGE_PATH}" \
        --container-name "${CONTAINER_NAME}" \
        --container-mounts "${STAGE_ROOT}:${STAGE_ROOT}" \
        --no-container-mount-home \
        sh -c "${direct_run_cmd}"
else
    srun -l \
        --nodes="${RUN_NNODES}" \
        --ntasks="${RUN_NNODES}" \
        --ntasks-per-node=1 \
        --gpus-per-node="${RUN_GPUS_PER_NODE}" \
        --mpi=none \
        --container-image "${IMAGE_PATH}" \
        --container-name "${CONTAINER_NAME}" \
        --container-mounts "${STAGE_ROOT}:${STAGE_ROOT}" \
        --no-container-mount-home \
        sh -c "${run_cmd}"
fi
