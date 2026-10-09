#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --job-name=coreai_comparch_sysarch-nep.cleanup-validation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --segment=1
#SBATCH --time=00:20:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/darfeen/Megatron-LM-nep-main-20260817}"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"

mkdir -p /home/darfeen/Megatron-LM/slurm_runs/lyris
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
        set -euo pipefail
        files=(
            examples/nonuniform/pretrain_gpt_nonuniform.py
            examples/nonuniform/pretrain_hybrid_nonuniform.py
            megatron/core/distributed/nonuniform_common.py
            megatron/core/distributed/nonuniform_ep.py
            megatron/core/optimizer/__init__.py
            megatron/core/transformer/moe/moe_layer.py
            megatron/core/transformer/moe/router.py
            megatron/core/transformer/moe/token_dispatcher.py
            megatron/core/transformer/transformer_config.py
            pretrain_hybrid.py
            tests/unit_tests/distributed/test_nonuniform_ep.py
        )
        python -m isort examples/nonuniform/pretrain_hybrid_nonuniform.py
        python -m black examples/nonuniform/pretrain_hybrid_nonuniform.py
        python -m ruff check "${files[@]}"
        python -m py_compile "${files[@]}"
        python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py
    '
