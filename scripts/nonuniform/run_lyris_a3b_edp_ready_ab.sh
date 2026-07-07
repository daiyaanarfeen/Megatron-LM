#!/bin/bash

# Same-allocation a3b comparison for the NEP EDP owner-readiness gate.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=lyris_a3b_edp_ready_ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_edp_ready_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"

run_case() {
    local gate="$1"
    local case_name="$2"

    echo "[a3b-edp-ready-ab] starting ${case_name} gate=${gate}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}" \
        IMAGE="${IMAGE}" \
        NAME="${case_name}" \
        TRAIN_ITERS=12 \
        LR_WSD_DECAY_ITERS=6 \
        NONUNIFORM_EP_TOPOLOGY="4 2" \
        TENSOR_MODEL_PARALLEL_SIZE=2 \
        EXPERT_MODEL_PARALLEL_SIZE=8 \
        EXPERT_TENSOR_PARALLEL_SIZE=1 \
        MICRO_BATCH_SIZE=4 \
        GLOBAL_BATCH_SIZE=48 \
        PROFILE=1 \
        PROFILE_STEP_START=5 \
        PROFILE_STEP_END=7 \
        PROFILE_RANKS="0 4 8" \
        HIGH_PRIORITY_STREAM_GROUPS="" \
        CUDA_DEVICE_MAX_CONNECTIONS=32 \
        NCCL_LAUNCH_ORDER_IMPLICIT=0 \
        MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1 \
        MEGATRON_NONUNIFORM_EP_EDP_READY_GATE="${gate}" \
        MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
        MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
        bash "${RUNNER}"
}

run_case 0 a3b_edp_ready_off
run_case 1 a3b_edp_ready_on
