#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.flex-callshape-direct-ep8-abba
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=3
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

HARNESS_ROOT="/home/darfeen/Megatron-LM"
REFERENCE_ROOT="/home/darfeen/Megatron-LM-nep-additive-ref"
CANDIDATE_ROOT="/home/darfeen/Megatron-LM-nep-flex-isolate"
RUNNER="${HARNESS_ROOT}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
DIRECT_RANK_LAUNCHER="${HARNESS_ROOT}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh"
RESULT_ROOT="${CANDIDATE_ROOT}/slurm_runs/nep_flex_callshape_direct_ep8_abba"
IMAGE="nvcr.io#nvidia/nemo:26.06"
TRAIN_ITERS=40
PROFILE_STEP_START=38
PROFILE_STEP_END=40
HYBRID_LAYER_PATTERN="MEMEM*EMEMEM*E"

mkdir -p "${RESULT_ROOT}" "${HARNESS_ROOT}/slurm_runs/lyris"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 3)); then
    echo "Expected three allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
FIRST_NODE="${allocated_nodes[0]}"
PROFILE_RANKS=$(seq -s ' ' 0 11)

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0
export MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0
export MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1
export MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1
export MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0
export MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0
export MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1
export MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0
export MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1
export MEGATRON_NONUNIFORM_EP_PARALLEL_GATHER_WINDOW=1
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=64
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=5
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter

run_case() {
    local label="$1"
    local source_root="$2"
    local entrypoint="$3"
    local master_port="$4"
    local callshape="$5"
    local name="a3b_repeat14_ep8_ep4_additive_${label}_distopt_i${TRAIN_ITERS}"
    local additional_container_mounts="${HARNESS_ROOT}:${HARNESS_ROOT}"
    if [[ "${source_root}" != "${CANDIDATE_ROOT}" ]]; then
        additional_container_mounts+=",${CANDIDATE_ROOT}:${CANDIDATE_ROOT}"
    fi

    echo "[additive-ep8] $(date --iso-8601=seconds) START ${label} source=${source_root}"
    timeout --foreground --signal=TERM --kill-after=45s 12m \
        env \
            MEGATRON_NONUNIFORM_EP_VALIDATED_FLEX_CALL_SHAPE="${callshape}" \
            REPO_DIR="${source_root}" \
            ROOT_DIR="${RESULT_ROOT}/${label}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="" \
            NAME="${name}" \
            MASTER_ADDR="${FIRST_NODE}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES=3 \
            RUN_WORLD_SIZE=12 \
            GPUS_PER_NODE=4 \
            NVLINK_SEGMENT_NODES=3 \
            USE_DIRECT_SRUN_RANKS=1 \
            DIRECT_RANK_LAUNCHER="${DIRECT_RANK_LAUNCHER}" \
            ADDITIONAL_CONTAINER_MOUNTS="${additional_container_mounts}" \
            PRETRAIN_ENTRYPOINT="${entrypoint}" \
            TRAIN_ITERS="${TRAIN_ITERS}" \
            LR_WSD_DECAY_ITERS=2 \
            HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN}" \
            NUM_EXPERTS=128 \
            MOE_ROUTER_TOPK=6 \
            SEQ_LENGTH=8192 \
            NONUNIFORM_MODE=ep \
            NONUNIFORM_EP_TOPOLOGY="4 2" \
            TENSOR_MODEL_PARALLEL_SIZE=2 \
            CONTEXT_PARALLEL_SIZE=1 \
            EXPERT_MODEL_PARALLEL_SIZE=8 \
            EXPERT_TENSOR_PARALLEL_SIZE=1 \
            MICRO_BATCH_SIZE=1 \
            GLOBAL_BATCH_SIZE=6 \
            TRUE_GLOBAL_BATCH_SIZE=6 \
            REPLICA_MICRO_BATCH_SIZES="1 1" \
            REPLICA_NUM_MICROBATCHES="1 1" \
            DDP_NUM_BUCKETS=8 \
            CUDA_GRAPH_IMPL=none \
            PROFILE=1 \
            PROFILE_STEP_START="${PROFILE_STEP_START}" \
            PROFILE_STEP_END="${PROFILE_STEP_END}" \
            PROFILE_RANKS="${PROFILE_RANKS}" \
            LOG_PARAMS_NORM=0 \
            LOG_NUM_ZEROS_IN_GRAD=0 \
            LOG_ENERGY=0 \
            MANUAL_GC_INTERVAL=1000 \
            LOG_INTERVAL=1 \
            HIGH_PRIORITY_STREAM_GROUPS=ep \
            MOE_ROUTER_FORCE_LOAD_BALANCING=0 \
            MOE_ROUTER_FORCE_UNIFORM_ROUTING=1 \
            MOE_ROUTER_ENABLE_EXPERT_BIAS=0 \
            NONUNIFORM_SKIP_OPTIMIZER_STEP=0 \
            DISTRIBUTED_TIMEOUT_MINUTES=5 \
            EXIT_DURATION_IN_MINS=11 \
            USE_GLOO_PROCESS_GROUPS=1 \
            SKIP_PREFLIGHT=1 \
            EXTRA_MEGATRON_ARGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend hybridep --calculate-per-token-loss --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --use-transformer-engine-op-fuser --moe-expert-rank-capacity-factor 1.5 --use-distributed-optimizer --overlap-param-gather --no-check-for-nan-in-loss-and-grad --rerun-mode disabled --nonuniform-disable-nongrad-sync-collectives" \
            bash "${RUNNER}"
    echo "[additive-ep8] $(date --iso-8601=seconds) DONE ${label}"
}

run_case warmup "${CANDIDATE_ROOT}" examples/nonuniform/pretrain_hybrid_nonuniform.py 31380 0
run_case baseline_a "${CANDIDATE_ROOT}" examples/nonuniform/pretrain_hybrid_nonuniform.py 31381 0
run_case callshape_a "${CANDIDATE_ROOT}" examples/nonuniform/pretrain_hybrid_nonuniform.py 31382 1
run_case callshape_b "${CANDIDATE_ROOT}" examples/nonuniform/pretrain_hybrid_nonuniform.py 31383 1
run_case baseline_b "${CANDIDATE_ROOT}" examples/nonuniform/pretrain_hybrid_nonuniform.py 31384 0

echo "[additive-ep8] complete: ${RESULT_ROOT}"
