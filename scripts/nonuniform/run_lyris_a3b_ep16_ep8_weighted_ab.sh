#!/bin/bash

# Same-allocation scale-up of the weighted EP8/EP4 benchmark to EP16/EP8.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep16-ep8-weighted-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep16_ep8_weighted_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_SELECTION="${CASE_SELECTION:-both}"
CASE_TIMEOUT="${CASE_TIMEOUT:-8m}"
CASE_PROFILE="${CASE_PROFILE:-1}"
REPEAT14_PATTERN="MEMEM*EMEMEM*E"

case "${CASE_SELECTION}" in
    both|healthy|nep) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}" >&2
        exit 2
        ;;
esac

preflight_done=0
case_index=0

run_case() {
    local name="$1"
    local mode="$2"
    local topology="$3"
    local run_nodes="$4"
    local world_size="$5"
    local global_batch_size="$6"
    local true_global_batch_size="$7"
    local replica_num_microbatches="$8"
    local split_host_phases=0
    local skip_preflight=1
    local profile_ranks
    local master_port

    if [[ "${mode}" == "ep" ]]; then
        split_host_phases=1
    fi
    if ((preflight_done == 0)); then
        skip_preflight=0
    fi
    profile_ranks="$(seq -s ' ' 0 "$((world_size - 1))")"
    master_port=$((30400 + case_index))
    case_index=$((case_index + 1))

    echo "[ep16-ep8-ab] $(date --iso-8601=seconds) starting ${name}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${name}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${run_nodes}" \
            RUN_WORLD_SIZE="${world_size}" \
            GPUS_PER_NODE=4 \
            USE_DIRECT_SRUN_RANKS=1 \
            TRAIN_ITERS=6 \
            LR_WSD_DECAY_ITERS=3 \
            HYBRID_LAYER_PATTERN="${REPEAT14_PATTERN}" \
            NUM_EXPERTS=128 \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=16 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE=2 \
            GLOBAL_BATCH_SIZE="${global_batch_size}" \
            TRUE_GLOBAL_BATCH_SIZE="${true_global_batch_size}" \
            REPLICA_MICRO_BATCH_SIZES="2 2" \
            REPLICA_NUM_MICROBATCHES="${replica_num_microbatches}" \
            DDP_NUM_BUCKETS=16 \
            CUDA_GRAPH_IMPL=local \
            PROFILE="${CASE_PROFILE}" \
            PROFILE_STEP_START=3 \
            PROFILE_STEP_END=4 \
            PROFILE_RANKS="${profile_ranks}" \
            EXTRA_MEGATRON_ARGS="--calculate-per-token-loss" \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=1 \
            TORCH_NCCL_BLOCKING_WAIT=0 \
            NCCL_DEBUG=WARN \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=0 \
            MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=3 \
            MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=4 \
            EXIT_DURATION_IN_MINS=7 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    preflight_done=1
    echo "[ep16-ep8-ab] $(date --iso-8601=seconds) completed ${name}"
}

if [[ "${CASE_SELECTION}" != "nep" ]]; then
    run_case \
        a3b_repeat14_ep16_ep16_mbs2_mb7_healthy \
        none "8 8" 8 32 224 224 "7 7"
fi

if [[ "${CASE_SELECTION}" != "healthy" ]]; then
    run_case \
        a3b_repeat14_ep16_ep8_mbs2_mb7_6_weighted \
        ep "8 4" 6 24 160 160 "7 6"
fi
