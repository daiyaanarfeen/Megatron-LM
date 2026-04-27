#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=3
#SBATCH --time=00:30:00
#SBATCH --job-name=bench_slot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
W=${WORKDIR:-$(pwd)}
I=${IMAGE:?Set IMAGE to the container image path}
M=$(scontrol show hostname $SLURM_NODELIST | head -n1)

HIDDEN="1024 2048 4096 8192"
# Bucket sizes for A: None (1 bucket) + several multi-bucket configs.
# Expert buf sizes: 16M, 67M, 268M, 1073M elements.
# bucket_size of 10M → ~2-107 buckets; 40M → 1-27 buckets; etc.
BUCKET_SIZES="10000000 40000000 100000000"
PORT=29500

for SLOT_MB in 128 256; do
    echo ""
    echo "############################################################"
    echo "# SLOT SIZE = ${SLOT_MB}MB"
    echo "############################################################"
    srun --container-image=$I --container-mounts=$W:$W --container-workdir=$W \
         --container-env=NCCL_NVLS_ENABLE=0,NVSHMEM_MAX_TEAMS=512,NVSHMEM_DISABLE_NVLS=1 \
         bash -c "
            export MEGATRON_NVSHMEM_SLOT_MB=$SLOT_MB
            pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null
            echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib > /etc/ld.so.conf.d/nvshmem.conf && ldconfig 2>/dev/null
            cd $W && pip install -e . --no-deps -q 2>/dev/null
            torchrun --nproc_per_node=4 --nnodes=\$SLURM_NNODES \
                --node_rank=\$SLURM_NODEID --master_addr=$M --master_port=$PORT \
                tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py \
                --hidden $HIDDEN \
                --bucket-sizes $BUCKET_SIZES 2>&1
         "
    PORT=$((PORT + 1))
done
