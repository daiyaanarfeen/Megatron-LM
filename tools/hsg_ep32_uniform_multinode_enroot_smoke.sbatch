#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=2
#SBATCH --time=00:15:00
#SBATCH --job-name=ep32_mn_smoke
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port}
LOGDIR=${LOGDIR:-$WORKDIR/ep32_uniform_multinode_smoke_$(date +%Y%m%d_%H%M%S)}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29712}
HSG_CACHED_IMAGE=${HSG_CACHED_IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-deepep-hybridep-megatron-nep-v3.sqsh}
ENROOT_ROOT=${ENROOT_ROOT:-/lustre/fsw/portfolios/coreai/users/darfeen/enroot}
ENROOT_CONTAINER_NAME=${ENROOT_CONTAINER_NAME:-megatron_pyt2510_hybridep_v3}

export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-$ENROOT_ROOT/cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-$ENROOT_ROOT/images}
export ENROOT_TEMP_PATH=${ENROOT_TEMP_PATH:-/tmp}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_LAUNCH_ORDER_IMPLICIT=${NCCL_LAUNCH_ORDER_IMPLICIT:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_3:1,mlx5_4:1}
export EP_NIC_NAME=${EP_NIC_NAME:-mlx5_0}
export EP_JIT_CACHE_DIR=${EP_JIT_CACHE_DIR:-$WORKDIR/.cache/deep_ep_jit}
export DEEP_EP_LD_PRELOAD=${DEEP_EP_LD_PRELOAD:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3}

cd "$WORKDIR"
mkdir -p "$LOGDIR" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_TEMP_PATH"

master=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
master=$(getent hosts "$master" | awk '{print $1}' | head -n1)
master=${master:-$(scontrol show hostname "$SLURM_NODELIST" | head -n1)}

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
  bash -lc "
    set -euo pipefail
    export ENROOT_CACHE_PATH='$ENROOT_CACHE_PATH'
    export ENROOT_DATA_PATH='$ENROOT_DATA_PATH'
    export ENROOT_TEMP_PATH='$ENROOT_TEMP_PATH'
    export NVIDIA_VISIBLE_DEVICES=\"\$CUDA_VISIBLE_DEVICES\"
    export NV_GPU=\"\$CUDA_VISIBLE_DEVICES\"
    export NVIDIA_DRIVER_CAPABILITIES=all
    node_container='${ENROOT_CONTAINER_NAME}_node'\$SLURM_NODEID
    if ! enroot list | awk '{print \$1}' | grep -Fxq \"\$node_container\"; then
      echo \"Creating cached Enroot container: \$node_container from $HSG_CACHED_IMAGE\"
      enroot create -n \"\$node_container\" '$HSG_CACHED_IMAGE'
    else
      echo \"Using cached Enroot container: \$node_container\"
    fi
    enroot start \
      --mount '$WORKDIR:$WORKDIR' \
      --env CUDA_VISIBLE_DEVICES \
      --env NVIDIA_VISIBLE_DEVICES \
      --env NV_GPU \
      --env NVIDIA_DRIVER_CAPABILITIES \
      --env SLURM_NODEID \
      --env NCCL_NVLS_ENABLE \
      --env NCCL_LAUNCH_ORDER_IMPLICIT \
      --env CUDA_DEVICE_MAX_CONNECTIONS \
      --env UCX_NET_DEVICES \
      --env PYTORCH_ALLOC_CONF \
      --env PYTORCH_CUDA_ALLOC_CONF \
      --env EP_NIC_NAME \
      --env EP_JIT_CACHE_DIR \
      --env DEEP_EP_LD_PRELOAD \
      --env MOE_EXPERT_CAPACITY_FACTOR \
      --env MOE_PAD_EXPERT_INPUT_TO_CAPACITY \
      \"\$node_container\" bash '$WORKDIR/tools/hsg_ep32_uniform_multinode_smoke_inside.sh' \
        '$master' '$MASTER_PORT' '$LOGDIR' '$GPUS_PER_NODE'
  "

echo "Multinode smoke log written under $LOGDIR"
