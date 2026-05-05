#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=1
#SBATCH --time=00:20:00
#SBATCH --job-name=prep_het_ep_img
#SBATCH --output=prep_het_ep_img_%j.out
#SBATCH --error=prep_het_ep_img_%j.err

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Prepare a reusable HSG Enroot/Pyxis image for the heterogeneous EP standard
# training comparison. This follows the HSG container docs' --container-save
# pattern: start from the base .sqsh once, install runtime Python dependencies,
# save a derived .sqsh, and use that derived image in future training jobs.

set -euo pipefail

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM}
BASE_IMAGE=${BASE_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.08-nvshmem.sqsh}
CACHED_IMAGE=${CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.08-nvshmem-megatron-het-ep.sqsh}

if [[ -f "$CACHED_IMAGE" && "${FORCE_REBUILD:-0}" != 1 ]]; then
  echo "Using existing cached image: $CACHED_IMAGE"
  exit 0
fi

if [[ -f "$CACHED_IMAGE" ]]; then
  echo "Refusing to overwrite existing cached image without removing it first: $CACHED_IMAGE"
  echo "Move or remove it manually, then rerun with FORCE_REBUILD=1."
  exit 1
fi

if [[ ! -f "$BASE_IMAGE" ]]; then
  echo "Missing base image: $BASE_IMAGE" >&2
  exit 1
fi

if [[ ! -d "$WORKDIR" ]]; then
  echo "Missing Megatron workdir: $WORKDIR" >&2
  exit 1
fi

echo "Base image: $BASE_IMAGE"
echo "Cached image: $CACHED_IMAGE"
echo "Workdir: $WORKDIR"

export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NVSHMEM_MAX_TEAMS=${NVSHMEM_MAX_TEAMS:-512}
export NVSHMEM_DISABLE_NVLS=${NVSHMEM_DISABLE_NVLS:-1}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-4G}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-save="$CACHED_IMAGE" \
  --container-mounts="$WORKDIR:$WORKDIR" \
  --container-workdir="$WORKDIR" \
  --container-env=NCCL_NVLS_ENABLE,NVSHMEM_MAX_TEAMS,NVSHMEM_DISABLE_NVLS \
  --container-env=NCCL_LAUNCH_ORDER_IMPLICIT,NVSHMEM_SYMMETRIC_SIZE,UCX_NET_DEVICES \
  bash -lc '
    set -euo pipefail
    ulimit -s 8192
    python -m pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12
    nvshmem_lib=$(python - <<'"'"'PY'"'"'
import pathlib
import site

for site_dir in site.getsitepackages():
    candidate = pathlib.Path(site_dir) / "nvidia" / "nvshmem" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PY
)
    if [[ -n "$nvshmem_lib" ]]; then
      echo "$nvshmem_lib" > /etc/ld.so.conf.d/nvshmem.conf
      ldconfig 2>/dev/null || true
    fi
    python -m pip install -e . --no-deps -q
    python - <<'"'"'PY'"'"'
import importlib.util
import torch

print("torch", torch.__version__)
print("megatron", importlib.util.find_spec("megatron") is not None)
print("nvshmem.core", importlib.util.find_spec("nvshmem.core") is not None)
PY
  '

ls -lh "$CACHED_IMAGE"
