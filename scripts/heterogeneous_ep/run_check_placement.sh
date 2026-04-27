#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=7
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=7
#SBATCH --time=00:10:00
#SBATCH --job-name=chk_place
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
W=${WORKDIR:-$(pwd)}
I=${IMAGE:?Set IMAGE to the container image path}
M=$(scontrol show hostname $SLURM_NODELIST | head -n1)

srun --container-image=$I --container-mounts=$W:$W --container-workdir=$W \
     --container-env=NCCL_NVLS_ENABLE=0,NVSHMEM_MAX_TEAMS=512,NVSHMEM_DISABLE_NVLS=1 \
     bash -c "
        pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null
        echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib > /etc/ld.so.conf.d/nvshmem.conf && ldconfig 2>/dev/null
        cd $W && pip install -e . --no-deps -q 2>/dev/null
        torchrun --nproc_per_node=4 --nnodes=\$SLURM_NNODES \
            --node_rank=\$SLURM_NODEID --master_addr=$M --master_port=29500 \
            $W/tools/heterogeneous_ep/check_placement.py 2>&1
     "
