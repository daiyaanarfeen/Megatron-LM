#!/bin/bash

# Seven-node reproduction of the EP64/48 first-backward deadlock. EP16/12
# with 48 experts preserves the three-versus-four local expert layout while
# making the launch-order control substantially cheaper.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.a3b48e-ep16-12-implicit

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep16_ep12_order_smoke}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_30b_moe_approach_a.sh"
NCCL_IMPLICIT_ORDER="${NCCL_IMPLICIT_ORDER:-1}"
CASE_TIMEOUT="${CASE_TIMEOUT:-18m}"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-10}"
NEP_DEBUG="${NEP_DEBUG:-1}"
NEP_DEBUG_RANKS="${NEP_DEBUG_RANKS:-0 12 16}"

echo "[a3b-ep16-12-order] $(date --iso-8601=seconds) implicit_order=${NCCL_IMPLICIT_ORDER}"
timeout --foreground --signal=TERM --kill-after=30s "${CASE_TIMEOUT}" \
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}" \
        IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}" \
        CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}" \
        NAME="a3b_48e_ep16_ep12_implicit${NCCL_IMPLICIT_ORDER}" \
        RUN_NNODES=7 \
        TRAIN_ITERS="${TRAIN_ITERS}" \
        LR_WSD_DECAY_ITERS=1 \
        NUM_EXPERTS=48 \
        NONUNIFORM_EP_TOPOLOGY="16 12" \
        TENSOR_MODEL_PARALLEL_SIZE=1 \
        EXPERT_MODEL_PARALLEL_SIZE=16 \
        EXPERT_TENSOR_PARALLEL_SIZE=1 \
        MICRO_BATCH_SIZE=2 \
        GLOBAL_BATCH_SIZE=56 \
        DDP_NUM_BUCKETS=16 \
        CUDA_GRAPH_IMPL=none \
        PROFILE=0 \
        HIGH_PRIORITY_STREAM_GROUPS="" \
        CUDA_DEVICE_MAX_CONNECTIONS=32 \
        NCCL_LAUNCH_ORDER_IMPLICIT="${NCCL_IMPLICIT_ORDER}" \
        MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1 \
        MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=1 \
        MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
        MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
        MEGATRON_NONUNIFORM_EP_DEBUG="${NEP_DEBUG}" \
        MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="${NEP_DEBUG_RANKS}" \
        DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES}" \
        EXIT_DURATION_IN_MINS=18 \
        SKIP_PREFLIGHT=1 \
        bash "${RUNNER}"

echo "[a3b-ep16-12-order] $(date --iso-8601=seconds) completed"
