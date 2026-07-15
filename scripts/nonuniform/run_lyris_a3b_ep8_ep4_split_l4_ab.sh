#!/bin/bash

# Four-layer representative a3b healthy/stable/split profiler comparison.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b-ep8-4-split-l4-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_split_l4_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-5m}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-7}"
CASE_SELECTION="${CASE_SELECTION:-all}"

case "${CASE_SELECTION}" in
    all|nep|healthy|stable|split) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}" >&2
        exit 2
        ;;
esac

run_case() {
    local name="$1"
    local mode="$2"
    local topology="$3"
    local nodes="$4"
    local global_batch_size="$5"
    local split_host_phases="$6"
    local master_port="$7"

    echo "[a3b-ep8-4-split-l4-ab] $(date --iso-8601=seconds) starting ${name}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${name}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${nodes}" \
            TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS=3 \
            HYBRID_LAYER_PATTERN=MEME \
            NUM_EXPERTS=128 \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            EXPERT_MODEL_PARALLEL_SIZE=8 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE=1 \
            GLOBAL_BATCH_SIZE="${global_batch_size}" \
            DDP_NUM_BUCKETS=16 \
            CUDA_GRAPH_IMPL=local \
            PROFILE=1 \
            PROFILE_STEP_START=3 \
            PROFILE_STEP_END=5 \
            PROFILE_RANKS="0 4 8" \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=1 \
            NCCL_DEBUG=WARN \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            MEGATRON_NONUNIFORM_EP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=3 \
            EXIT_DURATION_IN_MINS=4 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT=1 \
            bash "${RUNNER}"
    echo "[a3b-ep8-4-split-l4-ab] $(date --iso-8601=seconds) completed ${name}"
}

if [[ "${CASE_SELECTION}" == "all" || "${CASE_SELECTION}" == "healthy" ]]; then
    run_case a3b_l4_ep8_dp2_healthy none "4 4" 4 8 0 29941
fi
if [[ "${CASE_SELECTION}" == "all" || "${CASE_SELECTION}" == "nep" || "${CASE_SELECTION}" == "stable" ]]; then
    run_case a3b_l4_ep8_ep4_stable ep "4 2" 3 6 0 29942
fi
if [[ "${CASE_SELECTION}" == "all" || "${CASE_SELECTION}" == "nep" || "${CASE_SELECTION}" == "split" ]]; then
    run_case a3b_l4_ep8_ep4_split ep "4 2" 3 6 1 29943
fi
