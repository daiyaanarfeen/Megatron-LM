#!/bin/bash

# Small ordered-split numerical gate for multiple owner chunk counts.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300
#SBATCH --nodes=3
#SBATCH --segment=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-chunk-correct

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_ep8_ep4_chunk_correctness_sweep}"
RUNNER="${REPO_DIR}/scripts/nonuniform/run_lyris_ep8_ep4_split_correctness.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
TARGET_CHUNKS_LIST="${TARGET_CHUNKS_LIST:-2 4 8}"

srun --nodes=1 --ntasks=1 --mpi=none \
    --container-image="${IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${REPO_DIR}:${REPO_DIR}" \
    --container-workdir="${REPO_DIR}" \
    --no-container-mount-home \
    bash -lc "python -m torch.distributed.run --nproc-per-node=1 -m pytest -q \
        tests/unit_tests/distributed/test_nonuniform_ep.py \
        -k 'owner_layout_targets_balanced_chunks_with_byte_cap or one_target_has_identical_scheduler_inputs_to_original or owner_layout_rejects_nonpositive_target_chunks'"

for target_chunks in ${TARGET_CHUNKS_LIST}; do
    async_chunk_window=$((8 * target_chunks))
    if ((async_chunk_window < 16)); then
        async_chunk_window=16
    fi
    env \
        REPO_DIR="${REPO_DIR}" \
        ROOT_DIR="${ROOT_DIR}/chunks${target_chunks}" \
        IMAGE="${IMAGE}" \
        CONTAINER_NAME="${CONTAINER_NAME}" \
        CASE_TIMEOUT=6m \
        SPLIT_TARGET_CHUNKS="${target_chunks}" \
        SPLIT_ASYNC_CHUNK_WINDOW="${async_chunk_window}" \
        bash "${RUNNER}"
done
