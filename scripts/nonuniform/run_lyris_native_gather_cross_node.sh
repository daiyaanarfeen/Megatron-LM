#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200
#SBATCH --nodes=2
#SBATCH --segment=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --job-name=nep.native-gather-2n
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err

set -euo pipefail

REPO_DIR=/home/darfeen/Megatron-LM
IMAGE=nvcr.io#nvidia/nemo:26.06
CONTAINER_NAME=darfeen-nemo-2606
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0}
GROUP_RANKS=${GROUP_RANKS:-}
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 40000))
TRACE_DIR="${REPO_DIR}/slurm_runs/zero_cta/${SLURM_JOB_ID}"

GROUP_RANK_ARGS=""
if [[ -n "${GROUP_RANKS}" ]]; then
    GROUP_RANK_ARGS="--group-ranks ${GROUP_RANKS}"
fi
mkdir -p "${TRACE_DIR}"

run_cmd="cd ${REPO_DIR} && CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} CUDA_DEVICE_MAX_CONNECTIONS=1 TORCH_NCCL_AVOID_RECORD_STREAMS=1 NCCL_DEBUG=INFO python -u -m torch.distributed.run --nnodes=${SLURM_NNODES} --nproc-per-node=${GPUS_PER_NODE} --node-rank=\${SLURM_PROCID} --master-addr=${MASTER_ADDR} --master-port=${MASTER_PORT} scripts/nonuniform/probe_nccl_native_gather.py --trace-dir ${TRACE_DIR} --elements-per-rank 1048576 ${GROUP_RANK_ARGS}"

srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --mpi=none \
    --container-image="${IMAGE}" \
    --container-name="${CONTAINER_NAME}" \
    --container-mounts="${REPO_DIR}:${REPO_DIR}" \
    --container-workdir="${REPO_DIR}" \
    --no-container-mount-home \
    bash -lc "${run_cmd}"
