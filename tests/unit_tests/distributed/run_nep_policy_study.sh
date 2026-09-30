#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.policy-study
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export CODE_REPO=$base/.worktrees/nep-placement-controlled-20260929
export GRAPH_ALLOW_DIRTY=1 GRAPH_SCOPES=moe GRAPH_FULL_EP=16 GRAPH_REDUCED_EP=12 NCCL_PXN_DISABLE=0
mode=${STUDY_MODE:-replay}
[[ "$mode" == replay || "$mode" == training ]]
if [[ -z "${CLEANUP_CASES:-}" ]]; then
    CLEANUP_CASES=''
    for seed in ${STUDY_SEEDS:-101}; do
        for repeat in 0 1; do
            order='A E F C B D'
            [[ "$repeat" == 0 ]] || order='D B C F E A'
            for cell in $order; do
                CLEANUP_CASES+=" controlled_study_${cell}_s${seed}_r${repeat}_${mode}"
            done
        done
    done
fi
export CLEANUP_CASES
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base,/lustre:/lustre" bash -c '
    set -euo pipefail
    cd "$CODE_REPO"
    files=(tests/unit_tests/distributed/validate_nep_randomized_controls.py tests/unit_tests/distributed/validate_nep_study.py)
    uv run --no-project --active isort "${files[@]}"
    python -m black --skip-string-normalization "${files[@]}"
    '
exec bash "$base/scripts/nonuniform/run_nep_cleanup_ab.sh"
