#!/bin/bash

# Isolate host-only post-graph NEP progression in one allocation.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.post-graph-host-aba

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_a3b_ep8_ep4_post_graph_host_aba}"
SWEEP="${REPO_DIR}/scripts/nonuniform/run_lyris_a3b_ep8_ep4_batch_ratio_sweep.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_HYBRID_LAYER_PATTERN="${CASE_HYBRID_LAYER_PATTERN:-MEMEM*EMEMEM*E}"
CASE_INCLUDE_HEALTHY="${CASE_INCLUDE_HEALTHY:-1}"
CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS:-8}"
CASE_LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS:-4}"
CASE_TIMEOUT="${CASE_TIMEOUT:-7m}"

case "${CASE_INCLUDE_HEALTHY}" in
    0|1) ;;
    *)
        echo "CASE_INCLUDE_HEALTHY must be 0 or 1" >&2
        exit 2
        ;;
esac

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi
srun --nodes=1 --ntasks=1 --mpi=none "${container_args[@]}" \
    bash -lc "cd '${REPO_DIR}' && \
    python -m py_compile megatron/core/distributed/nonuniform_ep.py && \
    python -m pytest -q \
      tests/unit_tests/distributed/test_nonuniform_ep.py::test_nep_host_progress_registers_full_pipeline_after_non_moe_replay \
      tests/unit_tests/distributed/test_nonuniform_ep.py::test_nep_post_graph_host_progress_launches_full_pipeline_before_model_ep_fence \
      tests/unit_tests/distributed/test_nonuniform_ep.py::test_nep_deferred_post_graph_launches_full_pipeline_in_order"

preflight_done=0

run_case() {
    local label="$1"
    local selection="$2"
    local post_graph_host_phases="$3"
    local skip_preflight="${preflight_done}"

    echo "[post-graph-host-aba] $(date --iso-8601=seconds) starting ${label}"
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/${label}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_SELECTION="${selection}" \
        CASE_TIMEOUT="${CASE_TIMEOUT}" \
        CASE_PROFILE=1 \
        CASE_TRAIN_ITERS="${CASE_TRAIN_ITERS}" \
        CASE_LR_WSD_DECAY_ITERS="${CASE_LR_WSD_DECAY_ITERS}" \
        CASE_LOG_PARAMS_NORM=0 \
        CASE_LOG_NUM_ZEROS_IN_GRAD=0 \
        CASE_MANUAL_GC_INTERVAL=1000 \
        CASE_DEFER_MODEL_EP_FENCE=0 \
        CASE_SKIP_OWNER_GRAD_CHECK=0 \
        CASE_POST_GRAPH_HOST_PHASES="${post_graph_host_phases}" \
        CASE_HYBRID_LAYER_PATTERN="${CASE_HYBRID_LAYER_PATTERN}" \
        INITIAL_SKIP_PREFLIGHT="${skip_preflight}" \
        PAIR_MICRO_BATCH_SIZE=1 \
        PAIR_NUM_MICROBATCHES=1 \
        MOE_ROUTER_TOPK=6 \
        MOE_ROUTER_FORCE_LOAD_BALANCING=1 \
        MOE_ROUTER_FORCE_BIASED= \
        bash "${SWEEP}"
    preflight_done=1
    echo "[post-graph-host-aba] $(date --iso-8601=seconds) completed ${label}"
}

if [[ "${CASE_INCLUDE_HEALTHY}" == "1" ]]; then
    run_case healthy healthy 0
fi
run_case current_a proportional 0
run_case post_graph_host proportional 1
run_case current_a_repeat proportional 0
