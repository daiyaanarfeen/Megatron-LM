#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:06:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
export PROFILER_INSPECT_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/cupti_gpu_${SLURM_JOB_ID}
export CUPTI_TRACE_LIBRARY="${PROFILER_INSPECT_ROOT}/libgpu_trace.so"
mkdir -p "${PROFILER_INSPECT_ROOT}"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM,/lustre/fsw/coreai_comparch_sysarch/darfeen:/lustre/fsw/coreai_comparch_sysarch/darfeen \
    --no-container-mount-home bash -lc '
set -euo pipefail
sdk="${CUPTI_SDK_ROOT:-/usr/local/cuda}"
libdir="${CUPTI_SDK_LIBDIR:-${sdk}/lib64}"
headers="${CUPTI_CUDA_HEADERS:-/usr/local/cuda/include}"
g++ -std=c++17 -O2 -shared -fPIC -pthread -I"${sdk}/include" -I"${headers}" -I/usr/local/cuda/include \
    /home/darfeen/Megatron-LM/scripts/nonuniform/cupti_gpu_trace.cpp \
    -L"${libdir}" -Wl,--disable-new-dtags,-rpath,"${libdir}" -lcupti -o "${CUPTI_TRACE_LIBRARY}"
ldd "${CUPTI_TRACE_LIBRARY}"
if [[ -n "${CUPTI_SMOKE_GRAPH_MODES:-}" ]]; then
    root="${PROFILER_INSPECT_ROOT}"
    status=0
    for mode in ${CUPTI_SMOKE_GRAPH_MODES}; do
        echo "SMOKE_GRAPH_MODE=${mode}"
        mkdir -p "${root}/mode${mode}"
        CUPTI_WHOLE_GRAPH="${mode}" PROFILER_INSPECT_ROOT="${root}/mode${mode}" \
            python /home/darfeen/Megatron-LM/scripts/nonuniform/cupti_gpu_smoke.py || status=$?
    done
    exit "${status}"
else
    python /home/darfeen/Megatron-LM/scripts/nonuniform/cupti_gpu_smoke.py
fi
'
