#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:28:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.tail-control
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-param-prefetch-20260915
reference=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE="$reference"
unset NEP_SYNC_SHADOW
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 OMP_NUM_THREADS=1
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=6m
export SOFT_CLIQUE_MBS=2 SYNC_WORKLOAD_VARIANT=e512_seq2048
export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=40 SOFT_CLIQUE_TIMING_END=49 NEP_PREFETCH_VALIDATE=0
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 --no-container-mount-home \
 --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM \
 bash -lc "set -e; make -C '$candidate/megatron/core/datasets'; python -m py_compile /home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_tail_control.py"
for mode in sparse none full; do
    for case_name in healthy nep; do
        export NEP_TAIL_MODE="$mode" SOFT_CLIQUE_TAG="${mode}_${case_name}" CASE_ORDER="$case_name"
        export PARAM_TRANSFER_CODE_REPO="$candidate"
        if [[ "$case_name" == healthy ]]; then export PARAM_TRANSFER_CODE_REPO="$reference"; fi
        export NEP_SYNC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_tail_control.py
        if [[ "$mode" == full ]]; then
            export NEP_SYNC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_param_prefetch_probe.py
        fi
        export NEP_PREFETCH_EXPECTED_SHA
        NEP_PREFETCH_EXPECTED_SHA=$(sha256sum "$PARAM_TRANSFER_CODE_REPO/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)
        echo "TAIL_CASE_BEGIN $(date --iso-8601=seconds) $mode $case_name $NEP_PREFETCH_EXPECTED_SHA"
        bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
        echo "TAIL_CASE_DONE $(date --iso-8601=seconds) $mode $case_name"
    done
done
