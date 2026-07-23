#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.uniform-router-unit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=1
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/darfeen/Megatron-LM}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

mkdir -p "${REPO_ROOT}/slurm_runs/lyris"
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
        python -m isort \
            megatron/core/transformer/moe/router.py \
            tests/unit_tests/transformer/moe/test_routers.py &&
        python -m pytest -q \
            tests/unit_tests/transformer/moe/test_routers.py::test_exact_uniform_routing_logits \
            tests/unit_tests/transformer/moe/test_routers.py::test_exact_uniform_routing_requires_divisible_assignments \
            tests/unit_tests/transformer/moe/test_routers.py::test_exact_uniform_routing_config_contract \
            tests/unit_tests/transformer/moe/test_routers.py::TestTop2Router::test_force_uniform_routing
    '
