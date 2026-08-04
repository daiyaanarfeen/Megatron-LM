#!/bin/bash

# Same-allocation 14-stage healthy EP16/EP16 versus NEP EP16/EP12 benchmark.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep16-12-scatter

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep16_ep12_scatter_progress_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
CORRECTNESS="${REPO_DIR}/scripts/nonuniform/run_lyris_ep8_ep4_split_correctness.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"
RUN_HEALTHY_TIMING="${RUN_HEALTHY_TIMING:-1}"
RUN_NEP_TIMING="${RUN_NEP_TIMING:-1}"
RUN_HEALTHY_PROFILE="${RUN_HEALTHY_PROFILE:-1}"
RUN_NEP_PROFILE="${RUN_NEP_PROFILE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-18432}"
NUM_EXPERTS="${NUM_EXPERTS:-144}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-6}"
FULL_MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE:-4}"
REDUCED_MICRO_BATCH_SIZE="${REDUCED_MICRO_BATCH_SIZE:-1}"
SCATTER_CHUNKS="${SCATTER_CHUNKS:-1}"
PARALLEL_GATHER_WINDOW="${PARALLEL_GATHER_WINDOW:-1}"
DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS:-16}"
NEP_EXPERT_BUCKET_GROUPS="${NEP_EXPERT_BUCKET_GROUPS:-8}"
TIMING_ITERS="${TIMING_ITERS:-12}"
PROFILE_ITERS="${PROFILE_ITERS:-8}"
CASE_TIMEOUT="${CASE_TIMEOUT:-12m}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEMEM*EMEMEM*E}"

for toggle in RUN_CORRECTNESS RUN_HEALTHY_TIMING RUN_NEP_TIMING RUN_HEALTHY_PROFILE RUN_NEP_PROFILE; do
    case "${!toggle}" in
        0|1) ;;
        *)
            echo "${toggle} must be 0 or 1" >&2
            exit 2
            ;;
    esac
done
for value in SEQ_LENGTH NUM_EXPERTS MOE_ROUTER_TOPK FULL_MICRO_BATCH_SIZE REDUCED_MICRO_BATCH_SIZE SCATTER_CHUNKS DDP_NUM_BUCKETS NEP_EXPERT_BUCKET_GROUPS; do
    if ! [[ "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value} must be a positive integer" >&2
        exit 2
    fi
done
if ((SEQ_LENGTH % 2 != 0)); then
    echo "SEQ_LENGTH must be divisible by TP=2" >&2
    exit 2
fi
local_sequence_length=$((SEQ_LENGTH / 2))
if ((local_sequence_length * FULL_MICRO_BATCH_SIZE * MOE_ROUTER_TOPK % NUM_EXPERTS != 0)); then
    echo "Full-replica tokens do not divide uniformly across NUM_EXPERTS" >&2
    exit 2
fi
if ((local_sequence_length * REDUCED_MICRO_BATCH_SIZE * MOE_ROUTER_TOPK % NUM_EXPERTS != 0)); then
    echo "Reduced-replica tokens do not divide uniformly across NUM_EXPERTS" >&2
    exit 2
fi

healthy_true_gbs=$((16 * FULL_MICRO_BATCH_SIZE))
nep_true_gbs=$((8 * FULL_MICRO_BATCH_SIZE + 6 * REDUCED_MICRO_BATCH_SIZE))

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

echo "[ep16-12-scatter] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[ep16-12-scatter] seq=${SEQ_LENGTH} experts=${NUM_EXPERTS} topk=${MOE_ROUTER_TOPK} full_mbs=${FULL_MICRO_BATCH_SIZE} reduced_mbs=${REDUCED_MICRO_BATCH_SIZE} healthy_gbs=${healthy_true_gbs} nep_gbs=${nep_true_gbs} scatter_chunks=${SCATTER_CHUNKS} ddp_num_buckets=${DDP_NUM_BUCKETS} nep_expert_bucket_groups=${NEP_EXPERT_BUCKET_GROUPS}"

# Populate the named Enroot cache on every allocated node before the A/B cases.
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc "cd '${REPO_DIR}' && python -c 'import torch, mamba_ssm, causal_conv1d; print(\"[ep16-12-scatter] container warm: ok\")'"

if [[ "${RUN_CORRECTNESS}" == "1" ]]; then
    echo "[ep16-12-scatter] $(date --iso-8601=seconds) starting two-layer correctness gate"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/correctness" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_LABEL=ep16_ep12 \
        CASE_DISPLAY=ep16-12 \
        CASE_RUN_NNODES=7 \
        CASE_RUN_NPROC_PER_NODE=4 \
        CASE_GLOBAL_BATCH_SIZE=28 \
        CASE_NUM_LAYERS=2 \
        CASE_NUM_EXPERTS=48 \
        CASE_TENSOR_MODEL_PARALLEL_SIZE=2 \
        CASE_EXPERT_MODEL_PARALLEL_SIZE=16 \
        CASE_NONUNIFORM_EP_TOPOLOGY="8 6" \
        CASE_DEBUG_RANKS="0 12 16 24" \
        CASE_EXPECTED_RANKS=28 \
        REFERENCE_SCATTER_CHUNKS=1 \
        SPLIT_SCATTER_CHUNKS="${SCATTER_CHUNKS}" \
        SPLIT_A2A_SCATTER_SCHEDULER=0 \
        SPLIT_END_ITERATION_SCATTER=1 \
        REFERENCE_EXPERT_BUCKET_GROUPS="${NEP_EXPERT_BUCKET_GROUPS}" \
        SPLIT_EXPERT_BUCKET_GROUPS="${NEP_EXPERT_BUCKET_GROUPS}" \
        SPLIT_BUCKET_READY_GATHER=1 \
        SPLIT_EDP_READY_GATE=0 \
        SPLIT_PARALLEL_GATHER_WINDOW="${PARALLEL_GATHER_WINDOW}" \
        SPLIT_ASYNC_CHUNK_WINDOW=64 \
        CASE_TIMEOUT=7m \
        bash "${CORRECTNESS}"
    echo "[ep16-12-scatter] $(date --iso-8601=seconds) correctness gate passed"
fi

preflight_done=0
case_index=0

run_case() {
    local stage="$1"
    local label="$2"
    local mode="$3"
    local topology="$4"
    local run_nodes="$5"
    local world_size="$6"
    local true_global_batch_size="$7"
    local replica_micro_batch_sizes="$8"
    local profile="$9"
    local train_iters="${10}"
    local profile_step_start="${11}"
    local profile_step_end="${12}"
    local run_selected
    local split_host_phases=0
    local skip_preflight=1
    local profile_ranks
    local master_port=$((30800 + case_index))
    case_index=$((case_index + 1))

    case "${stage}:${mode}" in
        timing:none) run_selected="${RUN_HEALTHY_TIMING}" ;;
        timing:ep) run_selected="${RUN_NEP_TIMING}" ;;
        profile:none) run_selected="${RUN_HEALTHY_PROFILE}" ;;
        profile:ep) run_selected="${RUN_NEP_PROFILE}" ;;
    esac
    if [[ "${run_selected}" != "1" ]]; then
        echo "[ep16-12-scatter] skipping ${stage}/${label}"
        return
    fi

    if [[ "${mode}" == "ep" ]]; then
        split_host_phases=1
    fi
    if ((preflight_done == 0)); then
        skip_preflight=0
    fi
    profile_ranks="$(seq -s ' ' 0 "$((world_size - 1))")"

    echo "[ep16-12-scatter] $(date --iso-8601=seconds) starting ${stage}/${label}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}/${stage}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${label}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${run_nodes}" \
            RUN_WORLD_SIZE="${world_size}" \
            GPUS_PER_NODE=4 \
            USE_DIRECT_SRUN_RANKS=1 \
            TRAIN_ITERS="${train_iters}" \
            LR_WSD_DECAY_ITERS="$((train_iters / 2))" \
            HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN}" \
            NUM_EXPERTS="${NUM_EXPERTS}" \
            MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK}" \
            SEQ_LENGTH="${SEQ_LENGTH}" \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=16 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE}" \
            GLOBAL_BATCH_SIZE="${true_global_batch_size}" \
            TRUE_GLOBAL_BATCH_SIZE="${true_global_batch_size}" \
            REPLICA_MICRO_BATCH_SIZES="${replica_micro_batch_sizes}" \
            REPLICA_NUM_MICROBATCHES="1 1" \
            DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS}" \
            CUDA_GRAPH_IMPL=local \
            PROFILE="${profile}" \
            PROFILE_STEP_START="${profile_step_start}" \
            PROFILE_STEP_END="${profile_step_end}" \
            PROFILE_RANKS="${profile_ranks}" \
            LOG_PARAMS_NORM=0 \
            LOG_NUM_ZEROS_IN_GRAD=0 \
            LOG_ENERGY=0 \
            MANUAL_GC_INTERVAL=1000 \
            EXTRA_MEGATRON_ARGS="--calculate-per-token-loss --moe-router-bias-update-rate 0.0" \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            CUDA_DEVICE_MAX_CONNECTIONS=32 \
            NCCL_LAUNCH_ORDER_IMPLICIT=1 \
            TORCH_NCCL_BLOCKING_WAIT=0 \
            NCCL_DEBUG=WARN \
            MOE_ROUTER_FORCE_LOAD_BALANCING=0 \
            MOE_ROUTER_FORCE_UNIFORM_ROUTING=1 \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1 \
            MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1 \
            MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0 \
            MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0 \
            MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1 \
            MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0 \
            MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1 \
            MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW="${PARALLEL_GATHER_WINDOW}" \
            MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
            MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS="${SCATTER_CHUNKS}" \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64 \
            MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${NEP_EXPERT_BUCKET_GROUPS}" \
            MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=0 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS=11 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    preflight_done=1
    echo "[ep16-12-scatter] $(date --iso-8601=seconds) completed ${stage}/${label}"
}

run_pair() {
    local stage="$1"
    local profile="$2"
    local train_iters="$3"
    local profile_step_start="$4"
    local profile_step_end="$5"

    run_case \
        "${stage}" \
        "a3b_repeat14_ep16_ep16_mbs${FULL_MICRO_BATCH_SIZE}_healthy" \
        none "8 8" 8 32 "${healthy_true_gbs}" \
        "${FULL_MICRO_BATCH_SIZE} ${FULL_MICRO_BATCH_SIZE}" \
        "${profile}" "${train_iters}" "${profile_step_start}" "${profile_step_end}"
    run_case \
        "${stage}" \
        "a3b_repeat14_ep16_ep12_mbs${FULL_MICRO_BATCH_SIZE}_${REDUCED_MICRO_BATCH_SIZE}_weighted" \
        ep "8 6" 7 28 "${nep_true_gbs}" \
        "${FULL_MICRO_BATCH_SIZE} ${REDUCED_MICRO_BATCH_SIZE}" \
        "${profile}" "${train_iters}" "${profile_step_start}" "${profile_step_end}"
}

run_pair timing 0 "${TIMING_ITERS}" 6 7
run_pair profile 1 "${PROFILE_ITERS}" 5 7

echo "[ep16-12-scatter] complete root=${ROOT_DIR}"
