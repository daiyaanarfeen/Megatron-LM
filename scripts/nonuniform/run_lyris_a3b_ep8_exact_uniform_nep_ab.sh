#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.exact-ep8-4-ab
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
ROOT_DIR="${ROOT_DIR:-${REPO_ROOT}/slurm_runs/lyris_a3b_ep8_exact_uniform_nep_ab}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

mkdir -p "${ROOT_DIR}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --mpi=none \
    --container-image="${IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${REPO_ROOT}:${REPO_ROOT}" \
    --container-workdir="${REPO_ROOT}" \
    --no-container-mount-home \
    bash -lc '
        python -m isort scripts/nonuniform/analyze_exact_uniform_nep_ab.py &&
        python -m py_compile scripts/nonuniform/analyze_exact_uniform_nep_ab.py
    '

run_case() {
    local label="$1"
    local selection="$2"
    local profile="$3"
    local train_iters="$4"
    local profile_start="$5"
    local profile_end="$6"

    echo "[$(date --iso-8601=seconds)] Starting ${label}: selection=${selection}, profile=${profile}"
    env \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        REPO_DIR="${REPO_ROOT}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION="${selection}" \
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
        CASE_EXTRA_MEGATRON_ARGS="--moe-router-bias-update-rate 0.0" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        CASE_HYBRID_LAYER_PATTERN='MEMEM*EMEMEM*E' \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING=0 \
        MOE_ROUTER_FORCE_UNIFORM_ROUTING=1 \
        INITIAL_SKIP_PREFLIGHT="${INITIAL_SKIP_PREFLIGHT:-0}" \
        bash "${SWEEP_SCRIPT}"

    INITIAL_SKIP_PREFLIGHT=1
    export INITIAL_SKIP_PREFLIGHT
    echo "[$(date --iso-8601=seconds)] Finished ${label}"
}

# ABBA ordering controls linear allocation drift.
run_case healthy_timing_1 healthy 0 100 30 31
run_case nep_timing_1 proportional 0 100 30 31
run_case nep_timing_2 proportional 0 100 30 31
run_case healthy_timing_2 healthy 0 100 30 31

# Profiles are diagnostic only and are excluded from timing statistics.
run_case healthy_profile healthy 1 50 30 46
run_case nep_profile proportional 1 50 30 46

echo "[$(date --iso-8601=seconds)] Exact-uniform healthy/NEP comparison complete"
