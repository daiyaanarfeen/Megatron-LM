#!/bin/bash

# Same-allocation graph comparison: a3b healthy EP8/DP2 versus NEP EP8/EP4.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b-ep8-4-graph-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_graph_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
CASE_TIMEOUT="${CASE_TIMEOUT:-12m}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-7}"
CASE_SELECTION="${CASE_SELECTION:-both}"
CASE_PROFILE="${CASE_PROFILE:-1}"
CASE_NEP_SKIP_PREFLIGHT="${CASE_NEP_SKIP_PREFLIGHT:-0}"
CASE_NEP_ZERO_SM="${CASE_NEP_ZERO_SM:-1}"
CASE_NEP_EDP_READY_GATE="${CASE_NEP_EDP_READY_GATE:-${CASE_NEP_ZERO_SM}}"
CASE_NEP_HOST_EDP_READY_GATE="${CASE_NEP_HOST_EDP_READY_GATE:-0}"
CASE_NEP_SAME_COMM_READY="${CASE_NEP_SAME_COMM_READY:-0}"
CASE_NEP_DEFER_HOST_LAUNCH="${CASE_NEP_DEFER_HOST_LAUNCH:-0}"
CASE_NEP_NAME="${CASE_NEP_NAME:-a3b_ep8_ep4_nep_graph}"
CASE_PROFILE_RANKS="${CASE_PROFILE_RANKS:-0}"
CASE_PROFILE_RANKS="${CASE_PROFILE_RANKS//:/ }"
CASE_MICRO_BATCH_SIZE="${CASE_MICRO_BATCH_SIZE:-1}"
CASE_NEP_GLOBAL_BATCH_SIZE="${CASE_NEP_GLOBAL_BATCH_SIZE:-$((6 * CASE_MICRO_BATCH_SIZE))}"
CASE_HEALTHY_GLOBAL_BATCH_SIZE="${CASE_HEALTHY_GLOBAL_BATCH_SIZE:-$((8 * CASE_MICRO_BATCH_SIZE))}"

case "${CASE_SELECTION}" in
    both|nep|healthy) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}" >&2
        exit 2
        ;;
esac

run_case() {
    local case_name="$1"
    local mode="$2"
    local topology="$3"
    local run_nodes="$4"
    local global_batch="$5"
    local zero_sm="$6"
    local skip_preflight="$7"

    echo "[a3b-ep8-4-graph-ab] $(date --iso-8601=seconds) starting ${case_name}"
    timeout --foreground --signal=TERM --kill-after=60s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${case_name}" \
            RUN_NNODES="${run_nodes}" \
            TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS=4 \
            NUM_EXPERTS=128 \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            EXPERT_MODEL_PARALLEL_SIZE=8 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE="${CASE_MICRO_BATCH_SIZE}" \
            GLOBAL_BATCH_SIZE="${global_batch}" \
            DDP_NUM_BUCKETS=16 \
            CUDA_GRAPH_IMPL=local \
            PROFILE="${CASE_PROFILE}" \
            PROFILE_STEP_START=3 \
            PROFILE_STEP_END=5 \
            PROFILE_RANKS="${CASE_PROFILE_RANKS}" \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=1 \
            NCCL_DEBUG=WARN \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD="${zero_sm}" \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE="${CASE_NEP_EDP_READY_GATE}" \
            MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE="${CASE_NEP_HOST_EDP_READY_GATE}" \
            MEGATRON_NONUNIFORM_EP_SAME_COMM_READY="${CASE_NEP_SAME_COMM_READY}" \
            MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH="${CASE_NEP_DEFER_HOST_LAUNCH}" \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            MEGATRON_NONUNIFORM_EP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS=11 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    echo "[a3b-ep8-4-graph-ab] $(date --iso-8601=seconds) completed ${case_name}"
}

if [[ "${CASE_SELECTION}" == "both" || "${CASE_SELECTION}" == "nep" ]]; then
    run_case \
        "${CASE_NEP_NAME}" \
        ep \
        "4 2" \
        3 \
        "${CASE_NEP_GLOBAL_BATCH_SIZE}" \
        "${CASE_NEP_ZERO_SM}" \
        "${CASE_NEP_SKIP_PREFLIGHT}"
fi
if [[ "${CASE_SELECTION}" == "both" || "${CASE_SELECTION}" == "healthy" ]]; then
    healthy_skip_preflight=1
    if [[ "${CASE_SELECTION}" == "healthy" ]]; then
        healthy_skip_preflight=0
    fi
    run_case a3b_ep8_dp2_healthy_graph none "4 4" 4 "${CASE_HEALTHY_GLOBAL_BATCH_SIZE}" 0 "${healthy_skip_preflight}"
fi
