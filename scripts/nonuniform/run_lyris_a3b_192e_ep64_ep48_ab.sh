#!/bin/bash

# Minimal same-allocation comparison for a 192-expert a3b model. Run NEP
# first so a failure releases the 32-node allocation before the healthy case.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=32
#SBATCH --segment=16
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b192e-ep64-48-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_192e_ep64_ep48_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
CASE_TIMEOUT="${CASE_TIMEOUT:-9m}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-6}"
CASE_LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS:-3}"
CASE_CUDA_GRAPH_IMPL="${CASE_CUDA_GRAPH_IMPL:-none}"
CASE_PROFILE_STEP_START="${CASE_PROFILE_STEP_START:-2}"
CASE_PROFILE_STEP_END="${CASE_PROFILE_STEP_END:-3}"
CASE_EXIT_DURATION_IN_MINS="${CASE_EXIT_DURATION_IN_MINS:-13}"
CASE_NAME_SUFFIX="${CASE_NAME_SUFFIX:-}"
CASE_SELECTION="${CASE_SELECTION:-both}"

case "${CASE_SELECTION}" in
    both|nep|healthy) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}; expected both, nep, or healthy" >&2
        exit 2
        ;;
esac

run_case() {
    local case_name="$1"
    local topology="$2"
    local run_nodes="$3"
    local global_batch="$4"
    local profile_ranks="$5"
    local skip_preflight="$6"

    echo "[a3b-192e-ab] $(date --iso-8601=seconds) starting ${case_name} topology=${topology} nodes=${run_nodes} gbs=${global_batch}"
    timeout --foreground --signal=TERM --kill-after=60s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${case_name}" \
            RUN_NNODES="${run_nodes}" \
            TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS}" \
            NUM_EXPERTS=192 \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=64 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE=2 \
            GLOBAL_BATCH_SIZE="${global_batch}" \
            DDP_NUM_BUCKETS=16 \
            CUDA_GRAPH_IMPL="${CASE_CUDA_GRAPH_IMPL}" \
            PROFILE=1 \
            PROFILE_STEP_START="${CASE_PROFILE_STEP_START}" \
            PROFILE_STEP_END="${CASE_PROFILE_STEP_END}" \
            PROFILE_RANKS="${profile_ranks}" \
            HIGH_PRIORITY_STREAM_GROUPS="" \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=0 \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1 \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=1 \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS="${CASE_EXIT_DURATION_IN_MINS}" \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    echo "[a3b-192e-ab] $(date --iso-8601=seconds) completed ${case_name}"
}

# Full-replica owner 0, full-replica follower 48, and reduced-replica owner 0.
if [[ "${CASE_SELECTION}" != "healthy" ]]; then
    run_case "a3b_192e_nep_ep64_ep48${CASE_NAME_SUFFIX}" "64 48" 28 224 "0 48 64" 1
fi

# Both matching expert-DP owners in the healthy two-replica control.
if [[ "${CASE_SELECTION}" != "nep" ]]; then
    run_case "a3b_192e_healthy_ep64_ep64${CASE_NAME_SUFFIX}" "64 64" 32 256 "0 64" 1
fi
