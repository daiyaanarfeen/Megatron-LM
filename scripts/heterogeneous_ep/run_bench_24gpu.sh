#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_24gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
IMAGE=${IMAGE:?Set IMAGE to the container image path}

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun --container-image=$IMAGE \
     --container-mounts="$WORKDIR:$WORKDIR" \
     --container-workdir=$WORKDIR \
     --container-env=NCCL_NVLS_ENABLE=0 \
     bash -c "
        cd $WORKDIR
        pip install -e . --no-deps -q 2>/dev/null
        torchrun \
            --nproc_per_node=4 \
            --nnodes=$SLURM_NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py 2>&1
     "
