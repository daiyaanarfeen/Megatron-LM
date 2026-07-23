#!/bin/bash

# Isolate the owner-DDP gradient check and per-boundary model-EP fence in one allocation.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=3
#SBATCH --segment=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.reshard-cause

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_reshard_cause_ab}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-7m}"

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi
srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" bash -lc \
    "cd '${REPO_DIR}' && \
    python -m py_compile megatron/core/distributed/nonuniform_ep.py && \
    python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py::test_nep_benchmark_skip_owner_grad_check_is_opt_in && \
    if python -c 'import isort' >/dev/null 2>&1; then python -m isort --check-only megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py; else echo '[reshard-cause] isort unavailable in container'; fi"


run_case() {
    local label="$1"
    local skip_owner_grad_check="$2"
    local defer_model_ep_fence="$3"
    local skip_preflight="$4"

    echo "[reshard-cause] $(date --iso-8601=seconds) starting ${label}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION=proportional \
        CASE_TIMEOUT="${CASE_TIMEOUT}" \
        CASE_PROFILE=1 \
        CASE_TRAIN_ITERS=10 \
        CASE_LR_WSD_DECAY_ITERS=4 \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_SKIP_OWNER_GRAD_CHECK="${skip_owner_grad_check}" \
        CASE_DEFER_MODEL_EP_FENCE="${defer_model_ep_fence}" \
        INITIAL_SKIP_PREFLIGHT="${skip_preflight}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_BIASED= \
        bash "${SWEEP}"
    echo "[reshard-cause] $(date --iso-8601=seconds) completed ${label}"
}

run_case fence_check 0 0 0
run_case fence_no_check 1 0 1
run_case deferred_no_check 1 1 1
run_case fence_no_check_repeat 1 0 1
