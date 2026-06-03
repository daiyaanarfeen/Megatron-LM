#!/bin/bash
#SBATCH --job-name=ntp-mamba
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --segment=2
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=hsg_profile_traces/ntp_mamba/%x_%j.out
#SBATCH --error=hsg_profile_traces/ntp_mamba/%x_%j.err

set -euo pipefail

HOST_REPO=${HOST_REPO:-${SLURM_SUBMIT_DIR}}
TRACE_ROOT=${TRACE_ROOT:-${HOST_REPO}/hsg_profile_traces/ntp_mamba}
DEV_REF=${DEV_REF:-upstream/dev}
if git -C "${HOST_REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  DEV_COMMIT=$(git -C "${HOST_REPO}" rev-parse "${DEV_REF}")
  SRC_DIR=${TRACE_ROOT}/src_${DEV_COMMIT}
else
  DEV_COMMIT=${DEV_COMMIT:-gitless-source}
  SRC_DIR=${HOST_REPO}
fi
IMAGE=${IMAGE:-/lustre/fsw/portfolios/coreai/users/darfeen/images/ntp_mamba/pytorch_26.02-py3_mamba_ssm_2.3.2.post1_causal_conv1d_1.6.2.post1.sqsh}
CONTAINER_REPO=${CONTAINER_REPO:-/opt/megatron-lm}

BASE_TP=${BASE_TP:-4}
REDUCED_TP=${REDUCED_TP:-2}
LAYERS=${LAYERS:-8}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
SEQ_LEN=${SEQ_LEN:-1024}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
REDUCED_MICRO_BATCH_SIZE=${REDUCED_MICRO_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
STEPS=${STEPS:-8}
WARMUP_STEPS=${WARMUP_STEPS:-2}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
MAMBA_HEAD_DIM=${MAMBA_HEAD_DIM:-64}
MAMBA_NUM_HEADS=${MAMBA_NUM_HEADS:-16}
MAMBA_NUM_GROUPS=${MAMBA_NUM_GROUPS:-4}
MAMBA_STATE_DIM=${MAMBA_STATE_DIM:-64}
PROFILE_RANKS_UNIFORM=${PROFILE_RANKS_UNIFORM:-0,4}
PROFILE_RANKS_NTP=${PROFILE_RANKS_NTP:-0,2,4,5}
RECORD_SHAPES=${RECORD_SHAPES:-1}

mkdir -p "${TRACE_ROOT}"
if git -C "${HOST_REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1 && [[ ! -f "${SRC_DIR}/.gitless-dev-src" ]]; then
  mkdir -p "${SRC_DIR}"
  git -C "${HOST_REPO}" archive "${DEV_REF}" | tar -x -C "${SRC_DIR}"
  mkdir -p "${SRC_DIR}/tools"
  cp "${HOST_REPO}/tools/ntp_mamba_profile.py" "${SRC_DIR}/tools/ntp_mamba_profile.py"
  cp "${HOST_REPO}/tools/verify_mamba_imports.py" "${SRC_DIR}/tools/verify_mamba_imports.py"
  touch "${SRC_DIR}/.gitless-dev-src"
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((29000 + SLURM_JOB_ID % 10000))
export MASTER_ADDR MASTER_PORT
export PYTHONPATH=${CONTAINER_REPO}:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

record_shapes_flag=""
if [[ "${RECORD_SHAPES}" == "1" ]]; then
  record_shapes_flag="--record-shapes"
fi

echo "=== preflight imports: image=${IMAGE} dev=${DEV_COMMIT} ==="
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --container-image="${IMAGE}" \
  --container-mounts="${SRC_DIR}:${CONTAINER_REPO},${TRACE_ROOT}:${CONTAINER_REPO}/hsg_profile_traces/ntp_mamba" \
  --container-workdir="${CONTAINER_REPO}" \
  --container-env=PYTHONPATH,CUDA_DEVICE_MAX_CONNECTIONS,TORCH_NCCL_ASYNC_ERROR_HANDLING,PYTORCH_ALLOC_CONF \
  python3 tools/verify_mamba_imports.py

run_case() {
  local mode=$1
  local nnodes=$2
  local trace_ranks=$3
  local nproc_per_node=$4
  local trace_dir="${CONTAINER_REPO}/hsg_profile_traces/ntp_mamba/${mode}_${SLURM_JOB_ID}"
  local host_trace_dir="${TRACE_ROOT}/${mode}_${SLURM_JOB_ID}"
  mkdir -p "${host_trace_dir}"
  echo "=== ${mode}: nnodes=${nnodes} trace_dir=${host_trace_dir} dev=${DEV_COMMIT} ==="
  srun --nodes="${nnodes}" --ntasks="${nnodes}" --ntasks-per-node=1 \
    --container-image="${IMAGE}" \
    --container-mounts="${SRC_DIR}:${CONTAINER_REPO},${TRACE_ROOT}:${CONTAINER_REPO}/hsg_profile_traces/ntp_mamba" \
    --container-workdir="${CONTAINER_REPO}" \
    --container-env=MASTER_ADDR,MASTER_PORT,PYTHONPATH,CUDA_DEVICE_MAX_CONNECTIONS,TORCH_NCCL_ASYNC_ERROR_HANDLING,PYTORCH_ALLOC_CONF \
    bash -lc "ulimit -s 8192; torchrun \
      --nnodes=${nnodes} \
      --nproc_per_node=${nproc_per_node} \
      --node_rank=\${SLURM_NODEID} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      tools/ntp_mamba_profile.py \
        --mode ${mode} \
        --base-tp ${BASE_TP} \
        --reduced-tp ${REDUCED_TP} \
        --layers ${LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --mamba-head-dim ${MAMBA_HEAD_DIM} \
        --mamba-num-heads ${MAMBA_NUM_HEADS} \
        --mamba-num-groups ${MAMBA_NUM_GROUPS} \
        --mamba-state-dim ${MAMBA_STATE_DIM} \
        --seq-len ${SEQ_LEN} \
        --micro-batch-size ${MICRO_BATCH_SIZE} \
        --reduced-micro-batch-size ${REDUCED_MICRO_BATCH_SIZE} \
        --grad-accum-steps ${GRAD_ACCUM_STEPS} \
        --steps ${STEPS} \
        --warmup-steps ${WARMUP_STEPS} \
        --vocab-size ${VOCAB_SIZE} \
        --trace-ranks ${trace_ranks} \
        --trace-dir ${trace_dir} \
        ${record_shapes_flag}" 2>&1 | tee "${host_trace_dir}/${mode}.log"
}

# Uniform baseline uses TP4 x DP2 on 8 GPUs. NTP uses packed TP2 + TP4 on 6 GPUs.
# The NTP local batches default to TP4 replica MB=2 and TP2 replica MB=1, keeping
# samples proportional to active TP capacity.
run_case uniform 2 "${PROFILE_RANKS_UNIFORM}" 4
MASTER_PORT=$((MASTER_PORT + 1))
run_case ntp 2 "${PROFILE_RANKS_NTP}" 3

echo "Logs and traces written under ${TRACE_ROOT}"
