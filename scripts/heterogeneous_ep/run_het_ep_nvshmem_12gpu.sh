#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --segment=3
#SBATCH --job-name=het_nvshmem
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

WORKDIR=${WORKDIR:-$(pwd)}
IMAGE=${IMAGE:?Set IMAGE to the container image path}

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

run_test() {
    local test_filter="$1"
    local port="$2"
    echo "=== Running: $test_filter ==="
    srun --container-image=$IMAGE \
         --container-mounts="$WORKDIR:$WORKDIR" \
         --container-workdir=$WORKDIR \
         --container-env=NCCL_NVLS_ENABLE=0,NVSHMEM_MAX_TEAMS=512,NVSHMEM_DISABLE_NVLS=1 \
         bash -c "
            cd $WORKDIR
            pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null
            pip install -e . --no-deps -q 2>/dev/null
            # Register nvshmem lib path with ldconfig so dlopen can find it.
            echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib > /etc/ld.so.conf.d/nvshmem.conf && ldconfig 2>/dev/null
            torchrun \
                --nproc_per_node=4 \
                --nnodes=$SLURM_NNODES \
                --node_rank=\$SLURM_NODEID \
                --master_addr=$MASTER_ADDR \
                --master_port=$port \
                -m pytest -xvs -p no:rerunfailures \
                tests/unit_tests/distributed/test_heterogeneous_ep_grad_sync.py \
                -k '$test_filter' 2>&1
         "
    echo "=== Done: $test_filter ==="
}

# Approach A regression check
run_test 'TestApproachA_12GPU and test_grad_sync_sum' 29500
run_test 'TestApproachA_12GPU and test_cross_rank_consistency' 29501

# Approach B correctness (NVSHMEM ring allreduce)
run_test 'TestApproachB_12GPU and test_grad_sync_sum' 29502
run_test 'TestApproachB_12GPU and test_cross_rank_consistency' 29503

# Benchmark A vs B (256MB slot for good K values)
echo "=== Running benchmark (256MB slot) ==="
srun --container-image=$IMAGE \
     --container-mounts="$WORKDIR:$WORKDIR" \
     --container-workdir=$WORKDIR \
     --container-env=NCCL_NVLS_ENABLE=0,NVSHMEM_MAX_TEAMS=512,NVSHMEM_DISABLE_NVLS=1 \
     bash -c "
        export MEGATRON_NVSHMEM_SLOT_MB=256
        cd $WORKDIR
        pip install -q nvidia-nvshmem-cu12 nvshmem4py-cu12 2>/dev/null
        echo /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib > /etc/ld.so.conf.d/nvshmem.conf && ldconfig 2>/dev/null
        pip install -e . --no-deps -q 2>/dev/null
        torchrun \
            --nproc_per_node=4 \
            --nnodes=$SLURM_NNODES \
            --node_rank=\$SLURM_NODEID \
            --master_addr=$MASTER_ADDR \
            --master_port=29510 \
            tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py \
            --hidden 1024 2048 4096 8192 2>&1
     "

echo "=== ALL DONE ==="
