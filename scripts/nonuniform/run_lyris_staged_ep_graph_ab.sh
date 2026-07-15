#!/bin/bash

# Same-allocation toy comparison for staged NEP scale validation.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.staged-graph-ab

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_staged_ep_graph_ab}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_nonuniform_ep_overlap_smoke.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
FULL_EP_SIZE="${FULL_EP_SIZE:-8}"
REDUCED_EP_SIZE="${REDUCED_EP_SIZE:-4}"
NUM_EXPERTS="${NUM_EXPERTS:-8}"
CASE_SELECTION="${CASE_SELECTION:-both}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-6}"
CASE_TIMEOUT="${CASE_TIMEOUT:-5m}"
CASE_PROFILE="${CASE_PROFILE:-0}"
PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
PROFILE_STEP_END="${PROFILE_STEP_END:-5}"
PROFILE_RANKS="${PROFILE_RANKS:-0 4 8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-2048}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
GPUS_PER_NODE=4

case "${CASE_SELECTION}" in
    both|nep|healthy) ;;
    *)
        echo "Unsupported CASE_SELECTION=${CASE_SELECTION}" >&2
        exit 2
        ;;
esac

if (( FULL_EP_SIZE <= REDUCED_EP_SIZE )); then
    echo "FULL_EP_SIZE must be greater than REDUCED_EP_SIZE" >&2
    exit 2
fi
if (( NUM_EXPERTS % FULL_EP_SIZE != 0 || NUM_EXPERTS % REDUCED_EP_SIZE != 0 )); then
    echo "NUM_EXPERTS must be divisible by both EP sizes" >&2
    exit 2
fi

nep_world_size=$((FULL_EP_SIZE + REDUCED_EP_SIZE))
healthy_world_size=$((2 * FULL_EP_SIZE))
nep_nodes=$(((nep_world_size + GPUS_PER_NODE - 1) / GPUS_PER_NODE))
healthy_nodes=$((healthy_world_size / GPUS_PER_NODE))
if (( healthy_world_size % GPUS_PER_NODE != 0 || nep_world_size % nep_nodes != 0 )); then
    echo "Selected EP sizes do not map evenly to four-GPU nodes" >&2
    exit 2
fi
nep_nproc_per_node=$((nep_world_size / nep_nodes))
if (( nep_nproc_per_node > GPUS_PER_NODE )); then
    echo "NEP requires ${nep_nproc_per_node} processes per node" >&2
    exit 2
fi
required_nodes="${healthy_nodes}"
if [[ "${CASE_SELECTION}" == "nep" ]]; then
    required_nodes="${nep_nodes}"
fi
if (( SLURM_NNODES != required_nodes )); then
    echo "Expected ${required_nodes} allocated nodes, got ${SLURM_NNODES}" >&2
    exit 2
fi

run_case() {
    local case_name="$1"
    local mode="$2"
    local topology="$3"
    local run_nodes="$4"
    local nproc_per_node="$5"
    local global_batch_size="$6"
    local zero_sm="$7"
    local master_port="$8"

    echo "[staged-ep-graph-ab] $(date --iso-8601=seconds) starting ${case_name}"
    timeout --foreground --signal=TERM --kill-after=30s "${CASE_TIMEOUT}" \
        env \
            REPO_DIR="${REPO_DIR}" \
            ROOT_DIR="${ROOT_DIR}" \
            IMAGE="${IMAGE}" \
            CONTAINER_NAME="${CONTAINER_NAME}" \
            NAME="${case_name}_${SLURM_JOB_ID}" \
            MASTER_PORT="${master_port}" \
            RUN_NNODES="${run_nodes}" \
            RUN_NPROC_PER_NODE="${nproc_per_node}" \
            RUN_PREFLIGHT_TESTS=0 \
            ENABLE_PYTORCH_PROFILER="${CASE_PROFILE}" \
            PROFILE_STEP_START="${PROFILE_STEP_START}" \
            PROFILE_STEP_END="${PROFILE_STEP_END}" \
            PROFILE_RANKS="${PROFILE_RANKS}" \
            TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
            GLOBAL_BATCH_SIZE="${global_batch_size}" \
            MICRO_BATCH_SIZE=1 \
            NUM_LAYERS="${NUM_LAYERS}" \
            HIDDEN_SIZE="${HIDDEN_SIZE}" \
            FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE}" \
            NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS}" \
            SEQ_LENGTH="${SEQ_LENGTH}" \
            NUM_EXPERTS="${NUM_EXPERTS}" \
            EXPERT_MODEL_PARALLEL_SIZE="${FULL_EP_SIZE}" \
            NONUNIFORM_MODE="${mode}" \
            NONUNIFORM_EP_TOPOLOGY="${topology}" \
            MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD="${zero_sm}" \
            MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=1 \
            MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
            MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
            EXTRA_MEGATRON_ARGS="--cuda-graph-impl local --cuda-graph-modules moe_router --te-rng-tracker --no-load-rng --distributed-timeout-minutes 3" \
            bash "${RUNNER}"
    echo "[staged-ep-graph-ab] $(date --iso-8601=seconds) completed ${case_name}"
}

if [[ "${CASE_SELECTION}" == "both" || "${CASE_SELECTION}" == "nep" ]]; then
    run_case \
        "toy_e${NUM_EXPERTS}_ep${FULL_EP_SIZE}_ep${REDUCED_EP_SIZE}_nep_graph" \
        ep \
        "${FULL_EP_SIZE} ${REDUCED_EP_SIZE}" \
        "${nep_nodes}" \
        "${nep_nproc_per_node}" \
        "${nep_world_size}" \
        1 \
        29920
fi

if [[ "${CASE_SELECTION}" == "both" || "${CASE_SELECTION}" == "healthy" ]]; then
    run_case \
        "toy_e${NUM_EXPERTS}_ep${FULL_EP_SIZE}_dp2_healthy_graph" \
        none \
        "${FULL_EP_SIZE} ${FULL_EP_SIZE}" \
        "${healthy_nodes}" \
        "${GPUS_PER_NODE}" \
        "${healthy_world_size}" \
        0 \
        29921
fi
