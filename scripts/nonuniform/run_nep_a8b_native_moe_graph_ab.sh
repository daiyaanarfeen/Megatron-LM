#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.a8b-moe-graph
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
base=/home/darfeen/Megatron-LM
candidate="$base/.worktrees/nep-static-slot-capacity-20260921"
export PARAM_TRANSFER_CODE_REPO="$candidate" NEP_SYNC_REFERENCE="$candidate"
unset NEP_SYNC_SHADOW SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export NEP_SYNC_ENTRYPOINT="$base/scripts/nonuniform/pretrain_a8b_native_moe_graph.py"
export SOFT_CLIQUE_MBS=2 SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=cupti
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=65536 OMP_NUM_THREADS=1
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT=10m
export SOFT_CLIQUE_TRAIN_ITERS=40 SOFT_CLIQUE_PROFILE_START=25 SOFT_CLIQUE_PROFILE_END=35
export SOFT_CLIQUE_TIMING_START=25 SOFT_CLIQUE_TIMING_END=34
export A8B_DEPTH_EXPECTED_SHA=677332c72a1242ccd61334901c60b37c34500f021ecc96a527ffeec09b28966d
git -C "$candidate" diff --quiet HEAD
[[ "$(sha256sum "$candidate/megatron/core/distributed/nonuniform_ep.py" | cut -d' ' -f1)" == "$A8B_DEPTH_EXPECTED_SHA" ]]
srun --nodes=1 --ntasks=1 --mpi=none \
    --container-image=nvcr.io#nvidia/nemo:26.06 --container-name=nep_nemo_26_06 \
    --no-container-mount-home --container-mounts="$base:$base" \
    bash -lc "cd '$candidate'; make -C megatron/core/datasets"
for A8B_GRAPH_SCOPE in moe_router moe; do
    export A8B_GRAPH_SCOPE
    for CASE_ORDER in nep healthy; do
        export CASE_ORDER SOFT_CLIQUE_TAG="graph_${A8B_GRAPH_SCOPE}_${CASE_ORDER}"
        echo "A8B_GRAPH_BEGIN $(date --iso-8601=seconds) $SOFT_CLIQUE_TAG"
        # The legacy single-case summary defaults to random routing. Revalidate
        # completed training using its existing explicit uniform-routing mode.
        set +e
        bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
        status=$?
        set -e
        ((status == 0 || status == 1)) || exit "$status"
        python3 - "$base" "$SLURM_JOB_ID" "$SOFT_CLIQUE_TAG" "$CASE_ORDER" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/nonuniform'))
from training_scripts_nep_matrix import analyze_case, workload_by_name

job, tag, case = sys.argv[2:]
root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
run = root / job / tag / 'cupti/ep16/a8b_120b_latentmoe_1t' / case / job
w = workload_by_name(Path(sys.argv[1]), 'a8b_120b_latentmoe_1t')
result = analyze_case(w, case, run, 40, 25, 256 if case == 'healthy' else 224, 34, routing_mode='uniform')
(run / 'baseline_uniform.json').write_text(json.dumps(result, indent=2) + '\n')
print('UNIFORM_CASE_VALIDATED', case)
PY
        echo "A8B_GRAPH_DONE $(date --iso-8601=seconds) $SOFT_CLIQUE_TAG"
    done
done
