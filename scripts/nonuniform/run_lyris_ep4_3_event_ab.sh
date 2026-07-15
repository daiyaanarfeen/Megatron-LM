#!/bin/bash

# Same-allocation EP4/3 NEP versus standard EP4/DP2 healthy comparison.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:08:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep4-3-event-ab

set -euo pipefail

REPO_DIR=/home/darfeen/Megatron-LM
ROOT_DIR=${REPO_DIR}/slurm_runs/lyris_nep_smoke
RUNNER=${REPO_DIR}/scripts/nonuniform/run_lyris_nonuniform_ep_overlap_smoke.sh
AB_MICRO_BATCH_SIZE=${AB_MICRO_BATCH_SIZE:-1}
AB_NAME_SUFFIX=${AB_NAME_SUFFIX:-mbs${AB_MICRO_BATCH_SIZE}}
NEP_GLOBAL_BATCH_SIZE=${NEP_GLOBAL_BATCH_SIZE:-$((7 * AB_MICRO_BATCH_SIZE))}
HEALTHY_GLOBAL_BATCH_SIZE=${HEALTHY_GLOBAL_BATCH_SIZE:-$((8 * AB_MICRO_BATCH_SIZE))}
AB_TRAIN_ITERS=${AB_TRAIN_ITERS:-10}
AB_EXTRA_MEGATRON_ARGS=${AB_EXTRA_MEGATRON_ARGS:-}
AB_ENABLE_PYTORCH_PROFILER=${AB_ENABLE_PYTORCH_PROFILER:-0}
AB_NONUNIFORM_EP_DEBUG=${AB_NONUNIFORM_EP_DEBUG:-0}

common_env=(
    REPO_DIR=${REPO_DIR}
    ROOT_DIR=${ROOT_DIR}
    IMAGE=nvcr.io#nvidia/nemo:26.06
    CONTAINER_NAME=nep_nemo_26_06
    RUN_NNODES=3
    RUN_NPROC_PER_NODE=3
    USE_DIRECT_SRUN_RANKS=1
    MICRO_BATCH_SIZE=${AB_MICRO_BATCH_SIZE}
    NUM_EXPERTS=12
    TRAIN_ITERS=${AB_TRAIN_ITERS}
    NUM_LAYERS=16
    HIDDEN_SIZE=1024
    FFN_HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=16
    SEQ_LENGTH=1024
    RUN_PREFLIGHT_TESTS=0
    ENABLE_PYTORCH_PROFILER=${AB_ENABLE_PYTORCH_PROFILER}
    MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=${AB_NONUNIFORM_EP_DEBUG}
    MEGATRON_NONUNIFORM_EP_DEBUG=${AB_NONUNIFORM_EP_DEBUG}
    NCCL_LAUNCH_ORDER_IMPLICIT=1
)

env "${common_env[@]}" \
    NAME=ep4_3_ready_event_l16_unprofiled_ab_nep_${AB_NAME_SUFFIX} \
    RUN_WORLD_SIZE=7 \
    GLOBAL_BATCH_SIZE=${NEP_GLOBAL_BATCH_SIZE} \
    NONUNIFORM_EP_TOPOLOGY="4 3" \
    MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1 \
    MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=1 \
    MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
    EXTRA_MEGATRON_ARGS="--distributed-timeout-minutes 2 --ddp-num-buckets 4 ${AB_EXTRA_MEGATRON_ARGS}" \
    bash "${RUNNER}"

env "${common_env[@]}" \
    NAME=ep4_dp2_healthy_l16_unprofiled_ab_${AB_NAME_SUFFIX} \
    RUN_WORLD_SIZE=8 \
    GLOBAL_BATCH_SIZE=${HEALTHY_GLOBAL_BATCH_SIZE} \
    NONUNIFORM_EP_TOPOLOGY="4 4" \
    MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0 \
    EXTRA_MEGATRON_ARGS="--nonuniform-mode none --expert-model-parallel-size 4 --distributed-timeout-minutes 2 --ddp-num-buckets 4 ${AB_EXTRA_MEGATRON_ARGS}" \
    bash "${RUNNER}"
