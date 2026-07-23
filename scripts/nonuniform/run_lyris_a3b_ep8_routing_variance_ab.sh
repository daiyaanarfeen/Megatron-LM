#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-routing-var
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=4
#SBATCH --time=00:40:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/darfeen/Megatron-LM}"
SWEEP_SCRIPT="${REPO_ROOT}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/slurm_runs/lyris_a3b_ep8_routing_variance}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
ROUTER_BIAS_UPDATE_RATE="${ROUTER_BIAS_UPDATE_RATE:-0.0}"

mkdir -p "${ROOT_DIR}"

run_healthy() {
    local label="$1"
    local force_balance="$2"
    local profile="$3"
    local train_iters="$4"
    local profile_start="$5"
    local profile_end="$6"
    local profile_with_shapes="${7:-0}"
    local profile_with_stack="${8:-0}"
    local profile_extra_args=" --moe-router-bias-update-rate ${ROUTER_BIAS_UPDATE_RATE}"

    if [[ "${profile_with_shapes}" == "1" ]]; then
        profile_extra_args+=" --pytorch-profiler-collect-shapes"
    fi
    if [[ "${profile_with_stack}" == "1" ]]; then
        profile_extra_args+=" --pytorch-profiler-collect-callstack"
    fi

    echo "[$(date --iso-8601=seconds)] Starting ${label}: force_balance=${force_balance}, profile=${profile}, router_bias_update_rate=${ROUTER_BIAS_UPDATE_RATE}"
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
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_LOG_ENERGY=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_EXTRA_MEGATRON_ARGS="${profile_extra_args}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING="${force_balance}" \
        INITIAL_SKIP_PREFLIGHT="${INITIAL_SKIP_PREFLIGHT:-0}" \
        bash "${SWEEP_SCRIPT}"

    INITIAL_SKIP_PREFLIGHT=1
    export INITIAL_SKIP_PREFLIGHT
    echo "[$(date --iso-8601=seconds)] Finished ${label}"
}

# ABBA ordering controls for allocation drift while providing two independent
# profiler-free timing populations for each routing mode.
run_healthy balanced_timing_1 1 0 100 30 31
run_healthy natural_timing_1 0 0 100 30 31
run_healthy natural_timing_2 0 0 100 30 31
run_healthy balanced_timing_2 1 0 100 30 31

# Use a longer low-overhead profile for cross-iteration correlations, then two
# detailed steps per mode for host launch and Python-stack attribution. Start
# after iteration 30 so graph capture and early kernel-shape warmup are excluded.
run_healthy balanced_profile 1 1 50 30 46
run_healthy natural_profile 0 1 50 30 46
run_healthy balanced_profile_detail 1 1 32 30 32 1 1
run_healthy natural_profile_detail 0 1 32 30 32 1 1

echo "[$(date --iso-8601=seconds)] Routing variance characterization complete"
