#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --job-name=het_ep_24gpu
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
         --container-env=NCCL_NVLS_ENABLE=0 \
         bash -c "
            cd $WORKDIR
            pip install -e . --no-deps -q 2>/dev/null
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

run_test 'TestApproachA_24GPU and test_grad_sync_sum' 29500
run_test 'TestApproachA_24GPU and test_grad_sync_avg' 29501
run_test 'TestApproachA_24GPU and test_cross_rank' 29502
run_test 'TestApproachB_24GPU and test_grad_sync_sum' 29503
run_test 'TestApproachB_24GPU and test_grad_sync_avg' 29504
run_test 'TestApproachB_24GPU and test_cross_rank' 29505
run_test 'TestApproachB_24GPU and test_grad_sync_2_chunks' 29506

echo "=== ALL 24-GPU TESTS COMPLETE ==="
