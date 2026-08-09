#!/bin/bash

# Causal iteration-overhead decomposition for healthy EP8 and NEP EP8/EP4.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-phase-decomp

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_phase_decomposition}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_TIMING="${RUN_TIMING:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
TIMING_ITERS="${TIMING_ITERS:-10}"
PROFILE_ITERS="${PROFILE_ITERS:-8}"
CASE_TIMEOUT="${CASE_TIMEOUT:-7m}"
SEQ_LENGTH="${SEQ_LENGTH:-16384}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
FULL_MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE:-4}"
REDUCED_MICRO_BATCH_SIZE="${REDUCED_MICRO_BATCH_SIZE:-1}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEMEM*EMEMEM*E}"
DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS:-16}"
EXPERT_BUCKET_GROUPS="${EXPERT_BUCKET_GROUPS:-11}"

for toggle in RUN_PREFLIGHT RUN_TIMING RUN_PROFILE; do
    case "${!toggle}" in
        0|1) ;;
        *)
            echo "${toggle} must be 0 or 1" >&2
            exit 2
            ;;
    esac
done

healthy_true_gbs=$((8 * FULL_MICRO_BATCH_SIZE))
nep_true_gbs=$((4 * FULL_MICRO_BATCH_SIZE + 2 * REDUCED_MICRO_BATCH_SIZE))
manifest="${ROOT_DIR}/case_manifest_${SLURM_JOB_ID}.tsv"
mkdir -p "${ROOT_DIR}"
printf 'stage\tlabel\tmode\tphase_limit\tprofile\trun_nodes\tworld_size\ttrue_gbs\n' > "${manifest}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

echo "[phase-decomp] job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST}"
echo "[phase-decomp] healthy_gbs=${healthy_true_gbs} nep_gbs=${nep_true_gbs} timing_iters=${TIMING_ITERS} profile_iters=${PROFILE_ITERS}"

# Populate the named Enroot cache on each allocated node before timed cases.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" \
    bash -lc "cd '${REPO_DIR}' && python -c 'import torch, mamba_ssm, causal_conv1d; print(\"[phase-decomp] container warm: ok\")'"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
    srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" bash -lc "
        cd '${REPO_DIR}' &&
        python -m isort --check-only \
            megatron/core/distributed/nonuniform_ep.py \
            tests/unit_tests/distributed/test_nonuniform_ep.py &&
        python -m py_compile \
            megatron/core/distributed/nonuniform_ep.py \
            tests/unit_tests/distributed/test_nonuniform_ep.py &&
        python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py \
            -k 'benchmark_phase_limit or benchmark_truncated_contexts or two_level_gather_launches_one_native_edp'
    "
fi

preflight_done=0
case_index=0

run_case() {
    local stage="$1"
    local label="$2"
    local mode="$3"
    local phase_limit="$4"
    local profile="$5"
    local train_iters="$6"
    local run_nodes
    local world_size
    local topology
    local true_gbs
    local replica_mbs
    local split_host_phases
    local profile_ranks
    local skip_preflight=1
    local master_port=$((32000 + case_index))
    case_index=$((case_index + 1))

    if [[ "${mode}" == "none" ]]; then
        run_nodes=4
        world_size=16
        topology="4 4"
        true_gbs="${healthy_true_gbs}"
        replica_mbs="${FULL_MICRO_BATCH_SIZE} ${FULL_MICRO_BATCH_SIZE}"
        split_host_phases=0
    else
        run_nodes=3
        world_size=12
        topology="4 2"
        true_gbs="${nep_true_gbs}"
        replica_mbs="${FULL_MICRO_BATCH_SIZE} ${REDUCED_MICRO_BATCH_SIZE}"
        split_host_phases=1
    fi
    profile_ranks="$(seq -s ' ' 0 "$((world_size - 1))")"
    if ((preflight_done == 0)); then
        skip_preflight=0
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${stage}" "${label}" "${mode}" "${phase_limit}" "${profile}" \
        "${run_nodes}" "${world_size}" "${true_gbs}" >> "${manifest}"
    echo "[phase-decomp] $(date --iso-8601=seconds) starting ${stage}/${label} phase=${phase_limit}"
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
            NVLINK_SEGMENT_NODES=4 \
            USE_DIRECT_SRUN_RANKS=1 \
            TRAIN_ITERS="${train_iters}" \
            LR_WSD_DECAY_ITERS="$((train_iters / 2))" \
            HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN}" \
            NUM_EXPERTS="${NUM_EXPERTS}" \
            MOE_ROUTER_TOPK=6 \
            SEQ_LENGTH="${SEQ_LENGTH}" \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=8 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE="${FULL_MICRO_BATCH_SIZE}" \
            GLOBAL_BATCH_SIZE="${true_gbs}" \
            TRUE_GLOBAL_BATCH_SIZE="${true_gbs}" \
            REPLICA_MICRO_BATCH_SIZES="${replica_mbs}" \
            REPLICA_NUM_MICROBATCHES="1 1" \
            DDP_NUM_BUCKETS="${DDP_NUM_BUCKETS}" \
            CUDA_GRAPH_IMPL=local \
            PROFILE="${profile}" \
            PROFILE_STEP_START=5 \
            PROFILE_STEP_END=7 \
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
            MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES="${split_host_phases}" \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0 \
            MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1 \
            MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS= \
            MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1 \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${EXPERT_BUCKET_GROUPS}" \
            MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT="${phase_limit}" \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0 \
            MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK="${MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK:-1}" \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS=6 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT="${skip_preflight}" \
            bash "${RUNNER}"
    preflight_done=1
    echo "[phase-decomp] $(date --iso-8601=seconds) completed ${stage}/${label}"
}

if [[ "${RUN_TIMING}" == "1" ]]; then
    run_case timing healthy_pre none scatter 0 "${TIMING_ITERS}"
    for phase in none gather edp scatter; do
        run_case timing "nep_${phase}" ep "${phase}" 0 "${TIMING_ITERS}"
    done
    run_case timing healthy_post none scatter 0 "${TIMING_ITERS}"
fi

if [[ "${RUN_PROFILE}" == "1" ]]; then
    run_case profile healthy none scatter 1 "${PROFILE_ITERS}"
    run_case profile nep_scatter ep scatter 1 "${PROFILE_ITERS}"
fi

echo "[phase-decomp] complete root=${ROOT_DIR} manifest=${manifest}"
