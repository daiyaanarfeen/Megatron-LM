#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.buffer-unit
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=${CHECK_CODE_REPO:-$base/.worktrees/nep-buffer-cleanup-20260921}
export CHECK_ROOT=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_buffer_cleanup/$SLURM_JOB_ID
export PYTHONPATH="$CODE_REPO:$base/scripts/nonuniform:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=32 NCCL_LAUNCH_ORDER_IMPLICIT=0 NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS=1
mkdir -p "$CHECK_ROOT"
scontrol show job "$SLURM_JOB_ID" > "$CHECK_ROOT/allocation.txt"
srun --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    python -m black --skip-string-normalization megatron/core/distributed/nonuniform_ep.py tests/unit_tests/distributed/test_nonuniform_ep.py
    python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py > "$CHECK_ROOT/unit.log" 2>&1
    if [[ -f tests/unit_tests/distributed/nep_parameter_wave_smoke.py ]]; then
        uv run --no-project --active isort tests/unit_tests/distributed/nep_parameter_wave_smoke.py
        python -m black --skip-string-normalization tests/unit_tests/distributed/nep_parameter_wave_smoke.py
        export CUPTI_OUTPUT_DIR="$CHECK_ROOT/parameter_wave_correctness"
        timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_parameter_wave_smoke.py > "$CHECK_ROOT/parameter_wave.log" 2>&1
    fi
    if [[ -f tests/unit_tests/distributed/nep_parallel_owner_smoke.py ]]; then
        uv run --no-project --active isort tests/unit_tests/distributed/nep_parallel_owner_smoke.py
        python -m black --skip-string-normalization tests/unit_tests/distributed/nep_parallel_owner_smoke.py
        export CUPTI_OUTPUT_DIR="$CHECK_ROOT/parallel_owner_correctness"
        timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_parallel_owner_smoke.py > "$CHECK_ROOT/parallel_owner.log" 2>&1
    fi
    git diff > "$CHECK_ROOT/implementation.patch"
    export CUPTI_OUTPUT_DIR="$CHECK_ROOT/prefetch_correctness"
    timeout --signal=TERM --kill-after=10s 180 python -m torch.distributed.run --standalone --nproc-per-node=4 tests/unit_tests/distributed/nep_param_prefetch_smoke.py > "$CHECK_ROOT/prefetch.log" 2>&1
    for margin in ${DIAGNOSTIC_MARGINS-0 8 32}; do
    export NVTE_BWD_LAYERNORM_SM_MARGIN=$margin
    export DIAGNOSTIC_OUTPUT="$CHECK_ROOT/norm_interference_margin$margin"
    mkdir -p "$DIAGNOSTIC_OUTPUT"
    mkdir -p "$DIAGNOSTIC_OUTPUT/tmp"
    TMPDIR="$DIAGNOSTIC_OUTPUT/tmp" timeout --signal=TERM --kill-after=10s 180 nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite=true -o "$DIAGNOSTIC_OUTPUT/trace" python -m torch.distributed.run --standalone --nproc-per-node=2 /home/darfeen/Megatron-LM/scripts/nonuniform/repro_nccl_norm_interference.py > "$DIAGNOSTIC_OUTPUT/driver.log" 2>&1
    nsys export --type=sqlite --force-overwrite=true --output="$DIAGNOSTIC_OUTPUT/trace.sqlite" "$DIAGNOSTIC_OUTPUT/trace.nsys-rep"
    done
    echo BUFFER_CLEANUP_UNIT_AND_DIAGNOSTIC_DONE
    '
