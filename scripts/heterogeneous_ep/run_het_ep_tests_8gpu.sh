#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --job-name=het_ep_4gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euxo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
IMAGE=${IMAGE:?Set IMAGE to the container image path}

export NCCL_MAX_NCHANNELS=1
export NCCL_NVLS_ENABLE=0

srun --container-image=$IMAGE \
     --container-mounts="$WORKDIR:$WORKDIR" \
     --container-workdir=$WORKDIR \
     --container-env=NCCL_MAX_NCHANNELS=1,NCCL_NVLS_ENABLE=0 \
     bash -c "
        cd $WORKDIR
        pip install -e . --no-deps -q 2>/dev/null
        torchrun --nproc_per_node=4 -m pytest -xvs \
            tests/unit_tests/distributed/test_heterogeneous_ep_grad_sync.py -k 4GPU 2>&1
     "
