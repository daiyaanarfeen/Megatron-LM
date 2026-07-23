#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-exact-uniform
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
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/slurm_runs/lyris_a3b_ep8_exact_uniform}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
ROUTER_BIAS_UPDATE_RATE="${ROUTER_BIAS_UPDATE_RATE:-0.0}"

mkdir -p "${ROOT_DIR}"

run_healthy() {
    local label="$1"
    local routing_mode="$2"
    local profile="$3"
    local train_iters="$4"
    local profile_start="$5"
    local profile_end="$6"
    local force_random=0
    local force_uniform=0

    case "${routing_mode}" in
        random)
            force_random=1
            ;;
        exact)
            force_uniform=1
            ;;
        *)
            echo "Unsupported routing mode: ${routing_mode}" >&2
            exit 2
            ;;
    esac

    echo "[$(date --iso-8601=seconds)] Starting ${label}: routing=${routing_mode}, profile=${profile}"
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
        CASE_EXTRA_MEGATRON_ARGS="--moe-router-bias-update-rate ${ROUTER_BIAS_UPDATE_RATE}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING="${force_random}" \
        MOE_ROUTER_FORCE_UNIFORM_ROUTING="${force_uniform}" \
        INITIAL_SKIP_PREFLIGHT="${INITIAL_SKIP_PREFLIGHT:-0}" \
        bash "${SWEEP_SCRIPT}"

    INITIAL_SKIP_PREFLIGHT=1
    export INITIAL_SKIP_PREFLIGHT
    echo "[$(date --iso-8601=seconds)] Finished ${label}"
}

# ABBA timing order controls allocation drift.
run_healthy random_timing_1 random 0 100 30 31
run_healthy exact_timing_1 exact 0 100 30 31
run_healthy exact_timing_2 exact 0 100 30 31
run_healthy random_timing_2 random 0 100 30 31

# Profile enough stationary iterations on every rank to compare routing,
# participant readiness, and per-peer all-to-all split vectors.
run_healthy random_profile random 1 50 30 46
run_healthy exact_profile exact 1 50 30 46

echo "[$(date --iso-8601=seconds)] Exact-uniform routing A/B complete"
