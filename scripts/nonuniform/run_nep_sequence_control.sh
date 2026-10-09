#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:32:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.seq-control
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
candidate=/home/darfeen/Megatron-LM/.worktrees/nep-param-prefetch-20260915
reference=/home/darfeen/Megatron-LM/.worktrees/nep-sync-cpu-20260914
export NEP_SYNC_REFERENCE="$reference"
export NEP_SYNC_ENTRYPOINT=/home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_sequence_control.py
unset NEP_SYNC_SHADOW
export SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 OMP_NUM_THREADS=1
export WORKLOAD_FILTER=scaling_ladder/a770m_4b_moe_215b CASE_TIMEOUT=7m
export SOFT_CLIQUE_MBS=2 NEP_PREFETCH_VALIDATE=0
export SOFT_CLIQUE_TRAIN_ITERS=60 SOFT_CLIQUE_PROFILE_START=40 SOFT_CLIQUE_PROFILE_END=50
export SOFT_CLIQUE_TIMING_START=40 SOFT_CLIQUE_TIMING_END=49
srun --nodes=1 --ntasks=1 --mpi=none --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 --no-container-mount-home \
 --container-mounts=/home/darfeen/Megatron-LM:/home/darfeen/Megatron-LM \
 bash -lc "set -e; make -C '$candidate/megatron/core/datasets'; python -m py_compile /home/darfeen/Megatron-LM/scripts/nonuniform/pretrain_nep_sequence_control.py"
for sequence in 2048 8192; do
    export SYNC_WORKLOAD_VARIANT="e512_seq${sequence}" SOFT_CLIQUE_TAG="seq${sequence}"
    for case_name in healthy nep; do
        export CASE_ORDER="$case_name" PARAM_TRANSFER_CODE_REPO="$candidate"
        export NEP_PREFETCH_EXPECTED_SHA=91f23f619d898f4678dcd850c1f71a328c11f0ca663a60df046500248d6788a7
        if [[ "$case_name" == healthy ]]; then
            export PARAM_TRANSFER_CODE_REPO="$reference"
            export NEP_PREFETCH_EXPECTED_SHA=653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7
        fi
        actual_sha=$(sha256sum "$PARAM_TRANSFER_CODE_REPO/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)
        [[ "$actual_sha" == "$NEP_PREFETCH_EXPECTED_SHA" ]]
        echo "SEQUENCE_CASE_BEGIN $(date --iso-8601=seconds) $sequence $case_name $actual_sha"
        bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
        echo "SEQUENCE_CASE_DONE $(date --iso-8601=seconds) $sequence $case_name"
    done
done
