#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb300
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.delivery-gate
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-delivery-batching-20261002
export CONTROLLED_CODE_REPO=$CODE_REPO
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe
export GRAPH_FULL_EP=16 GRAPH_REDUCED_EP=12 NCCL_PXN_DISABLE=0
export NCCL_LAUNCH_ORDER_IMPLICIT=0 CLEANUP_CASE_TIMEOUT=15m CUDA_EVENT_POOL_SIZE=131072
if [[ ${NEP_STUDY_STAGE:-gate} == smoke ]]; then
    export GRAPH_FULL_EP=8 GRAPH_REDUCED_EP=6 NCCL_PXN_DISABLE=1
    export CLEANUP_CASES='controlled_delivery_deep controlled_delivery_consumer controlled_delivery_early'
else
    export CLEANUP_CASES=${CLEANUP_CASES:-'controlled_actual_M4_r0 controlled_actual_M1_r0_off controlled_actual_M1_r0'}
fi
files=(megatron/core/distributed/nonuniform_ep.py
       tests/unit_tests/distributed/nep_actual_ablation.py
       tests/unit_tests/distributed/nep_controlled_probe.py
       tests/unit_tests/distributed/nep_randomized_controls.py
       tests/unit_tests/distributed/nep_param_prefetch_smoke.py
       tests/unit_tests/distributed/validate_nep_work_allocation.py
       tests/unit_tests/distributed/validate_nep_study.py)
export STUDY_PYTHON_FILES="${files[*]}"
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    if [[ ${NEP_STUDY_FORMAT:-0} == 1 ]]; then
        uv run --no-project --active isort $STUDY_PYTHON_FILES
        python -m black --skip-string-normalization $STUDY_PYTHON_FILES
    fi
    python -m py_compile $STUDY_PYTHON_FILES
    '
root=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/$SLURM_JOB_ID
mkdir -p "$root/benchmark_source"
for file in "${files[@]}" tests/unit_tests/distributed/run_nep_delivery_ablation.sh tests/unit_tests/distributed/run_nep_delivery_cases.sh; do
    mkdir -p "$root/benchmark_source/$(dirname "$file")"
    cp "$CODE_REPO/$file" "$root/benchmark_source/$file"
done
bash "$CODE_REPO/tests/unit_tests/distributed/run_nep_delivery_cases.sh"
if [[ ${NEP_STUDY_STAGE:-gate} == smoke ]]; then
    PYTHONPATH="$CODE_REPO/tests/unit_tests/distributed" python3 - "$root" <<'PY'
import sys
from pathlib import Path
from validate_nep_study import audit
root = Path(sys.argv[1])
for run in sorted(root.glob('controlled_delivery_*')):
    audit(run, 8, 6, 'training', True)
    print('DELIVERY_TRAINING_SMOKE_PASS', run, flush=True)
PY
fi
