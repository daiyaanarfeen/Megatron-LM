#!/bin/bash

# Same-allocation EP16/EP12 expert-GEMM dimension A-B-A experiment.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep16-12-dim-gemm

set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/home/darfeen/Megatron-LM}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/darfeen/Megatron-LM-nep-main-20260817}"
ROOT_DIR="${ROOT_DIR:-${HARNESS_ROOT}/slurm_runs/lyris_a3b_ep16_ep12_dimension_gemm_aba}"
RUNNER="${HARNESS_ROOT}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
DIRECT_RANK_LAUNCHER="${HARNESS_ROOT}/scripts/nonuniform/run_lyris_a3b_direct_rank.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-9m}"
TRAIN_ITERS="${TRAIN_ITERS:-8}"
PROFILE_STEP_START="${PROFILE_STEP_START:-6}"
PROFILE_STEP_END="${PROFILE_STEP_END:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2016}"
NUM_EXPERTS="${NUM_EXPERTS:-144}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-6}"
HYBRID_LAYER_PATTERN="${HYBRID_LAYER_PATTERN:-MEME}"
BASE_HIDDEN_SIZE="${BASE_HIDDEN_SIZE:-2688}"
BASE_FFN_HIDDEN_SIZE="${BASE_FFN_HIDDEN_SIZE:-1856}"
HIGH_HIDDEN_SIZE="${HIGH_HIDDEN_SIZE:-5376}"
HIGH_FFN_HIDDEN_SIZE="${HIGH_FFN_HIDDEN_SIZE:-3712}"

for value in TRAIN_ITERS PROFILE_STEP_START PROFILE_STEP_END SEQ_LENGTH NUM_EXPERTS MOE_ROUTER_TOPK BASE_HIDDEN_SIZE BASE_FFN_HIDDEN_SIZE HIGH_HIDDEN_SIZE HIGH_FFN_HIDDEN_SIZE; do
    if ! [[ "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value} must be a positive integer" >&2
        exit 2
    fi
done
if ((PROFILE_STEP_START >= PROFILE_STEP_END || PROFILE_STEP_END > TRAIN_ITERS)); then
    echo "Profiler range must be nonempty and end no later than TRAIN_ITERS" >&2
    exit 2
fi
if ((NUM_EXPERTS % 16 != 0 || NUM_EXPERTS % 12 != 0)); then
    echo "NUM_EXPERTS must be divisible by both EP16 and EP12" >&2
    exit 2
fi
if ((SEQ_LENGTH % 2 != 0)); then
    echo "SEQ_LENGTH must be divisible by TP=2" >&2
    exit 2
fi
if ((((SEQ_LENGTH / 2) * MOE_ROUTER_TOPK) % NUM_EXPERTS != 0)); then
    echo "TP-local routed tokens must divide exactly across NUM_EXPERTS" >&2
    exit 2
fi

mkdir -p "${ROOT_DIR}" "${HARNESS_ROOT}/slurm_runs/lyris"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 8)); then
    echo "Expected eight allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
FIRST_NODE="${allocated_nodes[0]}"
PROFILE_RANKS="$(seq -s ' ' 0 27)"

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
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=8
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_MAX_GATHER_BYTES=8589934592
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_SCATTER=0
export MEGATRON_NONUNIFORM_EP_BENCHMARK_SKIP_OWNER_GRAD_CHECK=1
export MEGATRON_NONUNIFORM_EP_BENCHMARK_PHASE_LIMIT=scatter

warm_container_args=(
    "--container-image=${IMAGE}"
    "--container-mounts=${SOURCE_ROOT}:${SOURCE_ROOT},${HARNESS_ROOT}:${HARNESS_ROOT}"
    "--container-workdir=${SOURCE_ROOT}"
    "--no-container-mount-home"
    "--container-name=${CONTAINER_NAME}"
)
echo "[ep16-12-dim-gemm] $(date --iso-8601=seconds) warming named container on all nodes"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 --mpi=none "${warm_container_args[@]}" bash -lc "python -c 'import torch, mamba_ssm, causal_conv1d; print(\"container warm: ok\")'"

case_index=0
run_case() {
    local label="$1"
    local hidden_size="$2"
    local ffn_hidden_size="$3"
    local skip_preflight="$4"
    local name="a3b_ep16_ep12_dim_gemm_${label}_h${hidden_size}_f${ffn_hidden_size}"
    local master_port=$((31680 + case_index))
    case_index=$((case_index + 1))
    local -a case_env=(
        "REPO_DIR=${SOURCE_ROOT}"
        "ROOT_DIR=${ROOT_DIR}/${label}"
        "IMAGE=${IMAGE}"
        "CONTAINER_NAME=${CONTAINER_NAME}"
        "NAME=${name}"
        "MASTER_ADDR=${FIRST_NODE}"
        "MASTER_PORT=${master_port}"
        "RUN_NNODES=7"
        "RUN_WORLD_SIZE=28"
        "GPUS_PER_NODE=4"
        "NVLINK_SEGMENT_NODES=4"
        "USE_DIRECT_SRUN_RANKS=1"
        "DIRECT_RANK_LAUNCHER=${DIRECT_RANK_LAUNCHER}"
        "ADDITIONAL_CONTAINER_MOUNTS=${HARNESS_ROOT}:${HARNESS_ROOT}"
        "PRETRAIN_ENTRYPOINT=examples/nonuniform/pretrain_hybrid_nonuniform.py"
        "TRAIN_ITERS=${TRAIN_ITERS}"
        "LR_WSD_DECAY_ITERS=4"
        "HYBRID_LAYER_PATTERN=${HYBRID_LAYER_PATTERN}"
        "HIDDEN_SIZE=${hidden_size}"
        "FFN_HIDDEN_SIZE=${ffn_hidden_size}"
        "SHARED_EXPERT_INTERMEDIATE_SIZE=$((2 * ffn_hidden_size))"
        "NUM_EXPERTS=${NUM_EXPERTS}"
        "MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK}"
        "SEQ_LENGTH=${SEQ_LENGTH}"
        "NONUNIFORM_MODE=ep"
        "NONUNIFORM_EP_TOPOLOGY=8 6"
        "TENSOR_MODEL_PARALLEL_SIZE=2"
        "CONTEXT_PARALLEL_SIZE=1"
        "EXPERT_MODEL_PARALLEL_SIZE=16"
        "EXPERT_TENSOR_PARALLEL_SIZE=1"
        "MICRO_BATCH_SIZE=1"
        "GLOBAL_BATCH_SIZE=14"
        "TRUE_GLOBAL_BATCH_SIZE=14"
        "REPLICA_MICRO_BATCH_SIZES=1 1"
        "REPLICA_NUM_MICROBATCHES=1 1"
        "DDP_NUM_BUCKETS=8"
        "CUDA_GRAPH_IMPL=local"
        "PROFILE=1"
        "PROFILE_STEP_START=${PROFILE_STEP_START}"
        "PROFILE_STEP_END=${PROFILE_STEP_END}"
        "PROFILE_RANKS=${PROFILE_RANKS}"
        "LOG_PARAMS_NORM=0"
        "LOG_NUM_ZEROS_IN_GRAD=0"
        "LOG_ENERGY=0"
        "MANUAL_GC_INTERVAL=1000"
        "LOG_INTERVAL=1"
        "HIGH_PRIORITY_STREAM_GROUPS=ep"
        "MOE_ROUTER_FORCE_LOAD_BALANCING=0"
        "MOE_ROUTER_FORCE_UNIFORM_ROUTING=1"
        "MOE_ROUTER_ENABLE_EXPERT_BIAS=0"
        "NONUNIFORM_SKIP_OPTIMIZER_STEP=0"
        "DISTRIBUTED_TIMEOUT_MINUTES=5"
        "EXIT_DURATION_IN_MINS=0"
        "USE_GLOO_PROCESS_GROUPS=1"
        "SKIP_PREFLIGHT=${skip_preflight}"
        "EXTRA_MEGATRON_ARGS=--calculate-per-token-loss --moe-router-bias-update-rate 0.0 --no-check-for-nan-in-loss-and-grad --rerun-mode disabled --nonuniform-disable-nongrad-sync-collectives"
    )

    echo "[ep16-12-dim-gemm] $(date --iso-8601=seconds) START ${label} hidden=${hidden_size} expert_ffn=${ffn_hidden_size}"
    timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" env "${case_env[@]}" bash "${RUNNER}"
    echo "[ep16-12-dim-gemm] $(date --iso-8601=seconds) DONE ${label}"
}

run_case base_a "${BASE_HIDDEN_SIZE}" "${BASE_FFN_HIDDEN_SIZE}" 0
run_case high "${HIGH_HIDDEN_SIZE}" "${HIGH_FFN_HIDDEN_SIZE}" 1
run_case base_b "${BASE_HIDDEN_SIZE}" "${BASE_FFN_HIDDEN_SIZE}" 1

echo "[ep16-12-dim-gemm] complete root=${ROOT_DIR}"
