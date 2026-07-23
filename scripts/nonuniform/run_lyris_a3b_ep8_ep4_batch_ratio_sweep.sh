#!/bin/bash

# Same-allocation EP8/EP4 sweep around the proportional reduced-replica batch.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-batch-ratio

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_batch_ratio_sweep}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_SELECTION="${CASE_SELECTION:-all}"
CASE_TIMEOUT="${CASE_TIMEOUT:-10m}"
CASE_PROFILE="${CASE_PROFILE:-1}"
CASE_PROFILE_STEP_START="${CASE_PROFILE_STEP_START:-3}"
CASE_PROFILE_STEP_END="${CASE_PROFILE_STEP_END:-4}"
CASE_PROFILE_RANKS="${CASE_PROFILE_RANKS:-all}"
CASE_EXTRA_MEGATRON_ARGS="${CASE_EXTRA_MEGATRON_ARGS:-}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-8}"
CASE_LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS:-4}"
CASE_LOG_PARAMS_NORM="${CASE_LOG_PARAMS_NORM:-1}"
CASE_LOG_NUM_ZEROS_IN_GRAD="${CASE_LOG_NUM_ZEROS_IN_GRAD:-1}"
CASE_LOG_ENERGY="${CASE_LOG_ENERGY:-0}"
CASE_MANUAL_GC_INTERVAL="${CASE_MANUAL_GC_INTERVAL:-10}"
CASE_DEFER_MODEL_EP_FENCE="${CASE_DEFER_MODEL_EP_FENCE:-0}"
CASE_A2A_SCATTER_SCHEDULER="${CASE_A2A_SCATTER_SCHEDULER:-0}"
CASE_SKIP_OWNER_GRAD_CHECK="${CASE_SKIP_OWNER_GRAD_CHECK:-0}"
CASE_SEQ_LENGTH="${CASE_SEQ_LENGTH:-8192}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-6}"
INITIAL_SKIP_PREFLIGHT="${INITIAL_SKIP_PREFLIGHT:-0}"
PAIR_MICRO_BATCH_SIZE="${PAIR_MICRO_BATCH_SIZE:-2}"
PAIR_NUM_MICROBATCHES="${PAIR_NUM_MICROBATCHES:-7}"
PAIR_REDUCED_MICRO_BATCH_SIZE="${PAIR_REDUCED_MICRO_BATCH_SIZE:-${PAIR_MICRO_BATCH_SIZE}}"
PAIR_REDUCED_NUM_MICROBATCHES="${PAIR_REDUCED_NUM_MICROBATCHES:-${PAIR_NUM_MICROBATCHES}}"
REPEAT14_PATTERN="MEMEM*EMEMEM*E"
CASE_POST_GRAPH_HOST_PHASES="${CASE_POST_GRAPH_HOST_PHASES:-0}"
CASE_HYBRID_LAYER_PATTERN="${CASE_HYBRID_LAYER_PATTERN:-${REPEAT14_PATTERN}}"

case "${CASE_SELECTION}" in
    all|pair|healthy|below4|below5|below6|proportional|above8) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}" >&2
        exit 2
        ;;
esac

case_enabled() {
    if [[ "${CASE_SELECTION}" == "all" || "${CASE_SELECTION}" == "$1" ]]; then
        return 0
    fi
    [[ "${CASE_SELECTION}" == "pair" && ("$1" == "healthy" || "$1" == "proportional") ]]
}

case "${INITIAL_SKIP_PREFLIGHT}" in
    0|1) ;;
    *)
        echo "INITIAL_SKIP_PREFLIGHT must be 0 or 1" >&2
        exit 2
        ;;
esac
if ! [[ "${PAIR_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIR_MICRO_BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if ! [[ "${PAIR_NUM_MICROBATCHES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIR_NUM_MICROBATCHES must be a positive integer" >&2
    exit 2
fi
if ! [[ "${PAIR_REDUCED_MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIR_REDUCED_MICRO_BATCH_SIZE must be a positive integer" >&2
    exit 2
fi
if ! [[ "${PAIR_REDUCED_NUM_MICROBATCHES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIR_REDUCED_NUM_MICROBATCHES must be a positive integer" >&2
    exit 2
fi
if ! [[ "${CASE_SEQ_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CASE_SEQ_LENGTH must be a positive integer" >&2
    exit 2
fi
if ! [[ "${CASE_TRAIN_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CASE_TRAIN_ITERS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${CASE_PROFILE_STEP_START}" =~ ^[0-9]+$ ]] || \
    ! [[ "${CASE_PROFILE_STEP_END}" =~ ^[1-9][0-9]*$ ]] || \
    ((CASE_PROFILE_STEP_END <= CASE_PROFILE_STEP_START)); then
    echo "CASE_PROFILE_STEP_START/END must define a non-empty, non-negative interval" >&2
    exit 2
fi
if ! [[ "${CASE_LR_WSD_DECAY_ITERS}" =~ ^[1-9][0-9]*$ ]] || \
    ((CASE_LR_WSD_DECAY_ITERS > CASE_TRAIN_ITERS)); then
    echo "CASE_LR_WSD_DECAY_ITERS must be in [1, CASE_TRAIN_ITERS]" >&2
    exit 2
fi

preflight_done="${INITIAL_SKIP_PREFLIGHT}"
case_index=0
pair_healthy_gbs=$((8 * PAIR_MICRO_BATCH_SIZE * PAIR_NUM_MICROBATCHES))
pair_nep_gbs=$((4 * PAIR_MICRO_BATCH_SIZE * PAIR_NUM_MICROBATCHES + 2 * PAIR_REDUCED_MICRO_BATCH_SIZE * PAIR_REDUCED_NUM_MICROBATCHES))

run_case() {
    local name="$1"
    local mode="$2"
    local topology="$3"
    local run_nodes="$4"
    local world_size="$5"
    local true_global_batch_size="$6"
    local replica_num_microbatches="$7"
    local micro_batch_size="$8"
    local split_host_phases=0
    local post_graph_host_phases=0
    local replica_micro_batch_sizes="${9:-${micro_batch_size} ${micro_batch_size}}"
    local skip_preflight=1
    local profile_ranks
    local master_port

    if [[ "${mode}" == "ep" ]]; then
        split_host_phases=1
        post_graph_host_phases="${CASE_POST_GRAPH_HOST_PHASES}"
    fi
    if ((preflight_done == 0)); then
        skip_preflight=0
    fi
    if [[ "${CASE_PROFILE_RANKS}" == "all" ]]; then
        profile_ranks="$(seq -s ' ' 0 "$((world_size - 1))")"
    else
        profile_ranks="${CASE_PROFILE_RANKS}"
    fi
    master_port=$((30500 + case_index))
    case_index=$((case_index + 1))

    echo "[ep8-ep4-batch-ratio] $(date --iso-8601=seconds) starting ${name}"
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
            TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS}" \
            HYBRID_LAYER_PATTERN="${CASE_HYBRID_LAYER_PATTERN}" \
            NUM_EXPERTS=128 \
            MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK}" \
            SEQ_LENGTH="${CASE_SEQ_LENGTH}" \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=8 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE="${micro_batch_size}" \
            GLOBAL_BATCH_SIZE="${true_global_batch_size}" \
            TRUE_GLOBAL_BATCH_SIZE="${true_global_batch_size}" \
            REPLICA_MICRO_BATCH_SIZES="${replica_micro_batch_sizes}" \
            REPLICA_NUM_MICROBATCHES="${replica_num_microbatches}" \
            DDP_NUM_BUCKETS=16 \
            CUDA_GRAPH_IMPL=local \
            PROFILE="${CASE_PROFILE}" \
            PROFILE_STEP_START="${CASE_PROFILE_STEP_START}" \
            PROFILE_STEP_END="${CASE_PROFILE_STEP_END}" \
            PROFILE_RANKS="${profile_ranks}" \
            LOG_PARAMS_NORM="${CASE_LOG_PARAMS_NORM}" \
            LOG_NUM_ZEROS_IN_GRAD="${CASE_LOG_NUM_ZEROS_IN_GRAD}" \
            LOG_ENERGY="${CASE_LOG_ENERGY}" \
            MANUAL_GC_INTERVAL="${CASE_MANUAL_GC_INTERVAL}" \
            EXTRA_MEGATRON_ARGS="--calculate-per-token-loss ${CASE_EXTRA_MEGATRON_ARGS}" \
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
            MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE="${CASE_DEFER_MODEL_EP_FENCE}" \
            MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER="${CASE_A2A_SCATTER_SCHEDULER}" \
            MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES="${post_graph_host_phases}" \
            MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=3 \
            MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK="${CASE_SKIP_OWNER_GRAD_CHECK}" \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=4 \
            EXIT_DURATION_IN_MINS=9 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    preflight_done=1
    echo "[ep8-ep4-batch-ratio] $(date --iso-8601=seconds) completed ${name}"
}

if case_enabled healthy; then
    run_case \
        "a3b_repeat14_ep8_ep8_mbs${PAIR_MICRO_BATCH_SIZE}_mb${PAIR_NUM_MICROBATCHES}_healthy" \
        none "4 4" 4 16 "${pair_healthy_gbs}" \
        "${PAIR_NUM_MICROBATCHES} ${PAIR_NUM_MICROBATCHES}" "${PAIR_MICRO_BATCH_SIZE}" "${PAIR_MICRO_BATCH_SIZE} ${PAIR_MICRO_BATCH_SIZE}"
fi
if case_enabled below4; then
    run_case a3b_repeat14_ep8_ep4_mbs2_mb7_4_below ep "4 2" 3 12 72 "7 4" 2
fi
if case_enabled below5; then
    run_case a3b_repeat14_ep8_ep4_mbs2_mb7_5_below ep "4 2" 3 12 76 "7 5" 2
fi
if case_enabled below6; then
    run_case a3b_repeat14_ep8_ep4_mbs2_mb7_6_below ep "4 2" 3 12 80 "7 6" 2
fi
if case_enabled proportional; then
    run_case \
        "a3b_repeat14_ep8_ep4_mbs${PAIR_MICRO_BATCH_SIZE}_${PAIR_REDUCED_MICRO_BATCH_SIZE}_mb${PAIR_NUM_MICROBATCHES}_${PAIR_REDUCED_NUM_MICROBATCHES}_weighted" \
        ep "4 2" 3 12 "${pair_nep_gbs}" \
        "${PAIR_NUM_MICROBATCHES} ${PAIR_REDUCED_NUM_MICROBATCHES}" "${PAIR_MICRO_BATCH_SIZE}" "${PAIR_MICRO_BATCH_SIZE} ${PAIR_REDUCED_MICRO_BATCH_SIZE}"
fi
if case_enabled above8; then
    run_case a3b_repeat14_ep8_ep4_mbs2_mb7_8_above ep "4 2" 3 12 88 "7 8" 2
fi
