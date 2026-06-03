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
# pattern: start from a registry image or local .sqsh once, install runtime
# Python dependencies, save a derived .sqsh, and use that derived image in
# future training jobs.

set -euo pipefail

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM}
# PyTorch 25.10 ships TransformerEngine 2.8; Megatron requires TE >= 2.8
# for delay_wgrad_compute with overlap_grad_reduce.
BASE_IMAGE=${BASE_IMAGE:-nvcr.io#nvidia/pytorch:25.10-py3}
CACHED_IMAGE=${CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-nvshmem-megatron-het-ep.sqsh}
INSTALL_NVSHMEM=${INSTALL_NVSHMEM:-1}
NVSHMEM_PACKAGES=${NVSHMEM_PACKAGES:-"nvidia-nvshmem-cu12 nvshmem4py-cu12"}
EXTRA_PIP_PACKAGES=${EXTRA_PIP_PACKAGES:-pandas}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-10.0}
DISABLE_AGGRESSIVE_PTX_INSTRS=${DISABLE_AGGRESSIVE_PTX_INSTRS:-1}
HYBRID_EP_MULTINODE=${HYBRID_EP_MULTINODE:-0}
USE_NIXL=${USE_NIXL:-0}
MAX_JOBS=${MAX_JOBS:-16}
CUDA_CCCL_INCLUDE=${CUDA_CCCL_INCLUDE:-/usr/local/lib/python3.12/dist-packages/nvidia/cuda_cccl/include:/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl}

if [[ -f "$CACHED_IMAGE" && "${FORCE_REBUILD:-0}" != 1 ]]; then
  echo "Using existing cached image: $CACHED_IMAGE"
  exit 0
fi

if [[ -f "$CACHED_IMAGE" ]]; then
  echo "Refusing to overwrite existing cached image without removing it first: $CACHED_IMAGE"
  echo "Move or remove it manually, then rerun with FORCE_REBUILD=1."
  exit 1
fi

if [[ "$BASE_IMAGE" != *"#"* && ! -f "$BASE_IMAGE" ]]; then
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_CUDA_ARCH_LIST
export DISABLE_AGGRESSIVE_PTX_INSTRS
export HYBRID_EP_MULTINODE
export USE_NIXL
export MAX_JOBS
export CUDA_CCCL_INCLUDE
export CPATH="${CUDA_CCCL_INCLUDE}${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image="$BASE_IMAGE" \
  --container-save="$CACHED_IMAGE" \
  --container-mounts="$WORKDIR:$WORKDIR" \
  --container-workdir="$WORKDIR" \
  --container-env=NCCL_NVLS_ENABLE,NVSHMEM_MAX_TEAMS,NVSHMEM_DISABLE_NVLS \
  --container-env=NCCL_LAUNCH_ORDER_IMPLICIT,NVSHMEM_SYMMETRIC_SIZE,UCX_NET_DEVICES \
  --container-env=PYTORCH_CUDA_ALLOC_CONF \
  --container-env=TORCH_CUDA_ARCH_LIST,DISABLE_AGGRESSIVE_PTX_INSTRS \
  --container-env=HYBRID_EP_MULTINODE,USE_NIXL,MAX_JOBS \
  --container-env=CUDA_CCCL_INCLUDE,CPATH,CPLUS_INCLUDE_PATH \
  bash -lc '
    set -euo pipefail
    export TORCH_CUDA_ARCH_LIST="'"$TORCH_CUDA_ARCH_LIST"'"
    export DISABLE_AGGRESSIVE_PTX_INSTRS="'"$DISABLE_AGGRESSIVE_PTX_INSTRS"'"
    export HYBRID_EP_MULTINODE="'"$HYBRID_EP_MULTINODE"'"
    export USE_NIXL="'"$USE_NIXL"'"
    export MAX_JOBS="'"$MAX_JOBS"'"
    export CUDA_CCCL_INCLUDE="'"$CUDA_CCCL_INCLUDE"'"
    export CPATH="${CUDA_CCCL_INCLUDE}${CPATH:+:$CPATH}"
    export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    echo "Build env: TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST DISABLE_AGGRESSIVE_PTX_INSTRS=$DISABLE_AGGRESSIVE_PTX_INSTRS HYBRID_EP_MULTINODE=$HYBRID_EP_MULTINODE USE_NIXL=$USE_NIXL MAX_JOBS=$MAX_JOBS CUDA_CCCL_INCLUDE=$CUDA_CCCL_INCLUDE"
    ulimit -s 8192
    if [[ "'"$INSTALL_NVSHMEM"'" == "1" ]]; then
      python -m pip install -q '"$NVSHMEM_PACKAGES"'
    fi
    if [[ -n "'"$EXTRA_PIP_PACKAGES"'" ]]; then
      python -m pip install -q '"$EXTRA_PIP_PACKAGES"'
    fi
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
import importlib.metadata
import re
import torch

def version_tuple(version):
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()

te_version = importlib.metadata.version("transformer-engine")
if version_tuple(te_version) < (2, 8, 0):
    raise RuntimeError(f"TransformerEngine >= 2.8.0 required, found {te_version}")

print("torch", torch.__version__)
print("transformer-engine", te_version)
print("nccl", torch.cuda.nccl.version())
print("megatron", importlib.util.find_spec("megatron") is not None)
print("nvshmem.core", importlib.util.find_spec("nvshmem.core") is not None)
deep_ep_spec = importlib.util.find_spec("deep_ep")
print("deep_ep", deep_ep_spec is not None)
if deep_ep_spec is not None:
    import deep_ep
    print("deep_ep.Buffer", hasattr(deep_ep, "Buffer"))
    print("deep_ep.HybridEPBuffer", hasattr(deep_ep, "HybridEPBuffer"))
PY
  '

ls -lh "$CACHED_IMAGE"
