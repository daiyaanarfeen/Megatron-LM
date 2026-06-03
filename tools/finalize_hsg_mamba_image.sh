#!/bin/bash
#SBATCH --job-name=final-mamba-img
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=hsg_profile_traces/ntp_mamba/%x_%j.out
#SBATCH --error=hsg_profile_traces/ntp_mamba/%x_%j.err

set -euo pipefail

WORKDIR=${WORKDIR:-/lustre/fsw/portfolios/coreai/users/darfeen/ntp_mamba_dev_bd5c98f77_hsg}
IMAGE_DIR=${IMAGE_DIR:-/lustre/fsw/portfolios/coreai/users/darfeen/images/ntp_mamba}
CONTAINER_NAME=${CONTAINER_NAME:-ntp-mamba-pyt2602-2802367}
OUT_SQSH=${OUT_SQSH:-${IMAGE_DIR}/pytorch_26.02-py3_mamba_ssm_2.3.2.post1_causal_conv1d_1.6.2.post1.sqsh}

export ENROOT_CACHE_PATH=${ENROOT_CACHE_PATH:-${IMAGE_DIR}/enroot-cache}
export ENROOT_DATA_PATH=${ENROOT_DATA_PATH:-${IMAGE_DIR}/enroot-data}

mkdir -p "${IMAGE_DIR}" "${WORKDIR}/hsg_profile_traces/ntp_mamba"

echo "Verifying imports in rootfs ${CONTAINER_NAME}"
enroot start --rw --root \
  --mount "${WORKDIR}:${WORKDIR}" \
  --env PYTHONPATH="${WORKDIR}" \
  "${CONTAINER_NAME}" \
  python3 "${WORKDIR}/tools/verify_mamba_imports.py"

echo "Exporting ${OUT_SQSH}"
enroot export -f -o "${OUT_SQSH}" "${CONTAINER_NAME}"
ls -lh "${OUT_SQSH}"
echo "Cached image ready: ${OUT_SQSH}"
