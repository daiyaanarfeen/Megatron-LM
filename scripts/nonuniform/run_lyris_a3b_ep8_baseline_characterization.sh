#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-baseline-var
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=4
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/darfeen/Megatron-LM}"
SWEEP_SCRIPT="${REPO_ROOT}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/slurm_runs/lyris_a3b_ep8_baseline_characterization}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

mkdir -p "${ROOT_DIR}"

run_healthy() {
    local label="$1"
    local profile="$2"
    local train_iters="$3"
    local profile_start="$4"
    local profile_end="$5"
    local profile_with_shapes="${6:-0}"
    local profile_with_stack="${7:-0}"
    local profile_extra_args=""

    if [[ "${profile_with_shapes}" == "1" ]]; then
        profile_extra_args+=" --pytorch-profiler-collect-shapes"
    fi
    if [[ "${profile_with_stack}" == "1" ]]; then
        profile_extra_args+=" --pytorch-profiler-collect-callstack"
    fi

    echo "[$(date --iso-8601=seconds)] Starting ${label}: profile=${profile}, train_iters=${train_iters}"

    env \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        REPO_DIR="${REPO_ROOT}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=healthy \
        CASE_PROFILE="${profile}" \
        CASE_PROFILE_STEP_START="${profile_start}" \
        CASE_PROFILE_STEP_END="${profile_end}" \
        CASE_PROFILE_RANKS=all \
        CASE_TRAIN_ITERS="${train_iters}" \
        CASE_LR_WSD_DECAY_ITERS="$((train_iters / 2))" \
        CASE_TIMEOUT=9m \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_LOG_ENERGY=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_EXTRA_MEGATRON_ARGS="${profile_extra_args}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        INITIAL_SKIP_PREFLIGHT="${INITIAL_SKIP_PREFLIGHT:-0}" \
        bash "${SWEEP_SCRIPT}"

    INITIAL_SKIP_PREFLIGHT=1
    export INITIAL_SKIP_PREFLIGHT
    echo "[$(date --iso-8601=seconds)] Finished ${label}"
}

# Keep profiler overhead out of the timing population. Iterations 1-4 are excluded
# by the analyzer because they include graph capture and post-capture settling.
run_healthy timing_1 0 50 5 6
run_healthy timing_2 0 50 5 6
run_healthy timing_3 0 50 5 6

# Sixteen active profiler steps are enough to identify repeatable rank/operation
# effects without making the timing population itself profiler-dependent.
run_healthy profile 1 23 5 21

# Retain two detailed steps for host-side launch-gap and shape attribution. This
# is deliberately separate from both the baseline timings and the longer trace.
run_healthy profile_detail 1 7 5 7 1 1

echo "[$(date --iso-8601=seconds)] Baseline characterization complete"
