#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=7
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --segment=7
#SBATCH --time=00:20:00
#SBATCH --job-name=het_28gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
W=${WORKDIR:-$(pwd)}
I=${IMAGE:?Set IMAGE to the container image path}
M=$(scontrol show hostname $SLURM_NODELIST | head -n1)

run() {
    local desc=$1 port=$2
    shift 2
    echo "=== $desc ==="
    srun --container-image=$I --container-mounts=$W:$W --container-workdir=$W \
         --container-env=NCCL_NVLS_ENABLE=0,NVSHMEM_MAX_TEAMS=512,NVSHMEM_DISABLE_NVLS=1 \
         bash -c "
            export MEGATRON_NVSHMEM_SLOT_MB=256
            pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null
            echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib > /etc/ld.so.conf.d/nvshmem.conf && ldconfig 2>/dev/null
            cd $W && pip install -e . --no-deps -q 2>/dev/null
            torchrun --nproc_per_node=4 --nnodes=\$SLURM_NNODES \
                --node_rank=\$SLURM_NODEID --master_addr=$M --master_port=$port \
                $@ 2>&1
         "
    echo "=== Done: $desc ==="
}

# Integration training test (A vs B, 10 steps)
run "Training integration" 29500 \
    tests/unit_tests/distributed/test_heterogeneous_ep_training.py --hidden 64 --steps 10

# Benchmark A vs B
run "Benchmark" 29510 \
    tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py --hidden 1024 2048 4096
