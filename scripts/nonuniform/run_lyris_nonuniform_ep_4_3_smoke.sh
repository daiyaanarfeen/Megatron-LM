#!/bin/bash

# Seven-rank 4:3 NEP topology reproducer for non-integral replica ratios.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep4-3-smoke

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_nonuniform_ep_overlap_smoke.sh"

env \
    REPO_DIR="${REPO_DIR}" \
    ROOT_DIR="${REPO_DIR}/slurm_runs/lyris_nep_4_3_smoke" \
    IMAGE="nvcr.io#nvidia/nemo:26.06" \
    CONTAINER_NAME="nep_nemo_26_06" \
    NAME="ep4_3_eager_debug_${SLURM_JOB_ID}" \
    RUN_NNODES=2 \
    RUN_NPROC_PER_NODE=4 \
    RUN_WORLD_SIZE=7 \
    USE_DIRECT_SRUN_RANKS=1 \
    RUN_PREFLIGHT_TESTS=0 \
    ENABLE_PYTORCH_PROFILER=0 \
    TRAIN_ITERS=2 \
    GLOBAL_BATCH_SIZE=7 \
    MICRO_BATCH_SIZE=1 \
    NUM_LAYERS=4 \
    HIDDEN_SIZE=512 \
    FFN_HIDDEN_SIZE=2048 \
    NUM_ATTENTION_HEADS=8 \
    SEQ_LENGTH=512 \
    NUM_EXPERTS=12 \
    NONUNIFORM_EP_TOPOLOGY="4 3" \
    MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=1 \
    MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=1 \
    MEGATRON_NONUNIFORM_EP_NCCL_ASYNC_CHUNK_WINDOW=16 \
    MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0 \
    MEGATRON_NONUNIFORM_EP_DEBUG=1 \
    MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="0 3 4" \
    EXTRA_MEGATRON_ARGS="--distributed-timeout-minutes 3" \
    bash "${RUNNER}"
