#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.graph-attribution
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
base=/home/darfeen/Megatron-LM
candidate="${GRAPH_CODE_REPO:-$base/.worktrees/nep-graph-launch-order-20260921}"
export NEP_LAUNCH_ORDER_IMPLICIT=${NEP_LAUNCH_ORDER_IMPLICIT:-0}
export PARAM_TRANSFER_CODE_REPO="$candidate" NEP_SYNC_REFERENCE="$candidate"
unset NEP_SYNC_SHADOW SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export NEP_SYNC_ENTRYPOINT="$base/scripts/nonuniform/pretrain_a8b_graph_attribution.py"
export SOFT_CLIQUE_MBS=2 SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=nsys
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=32768 OMP_NUM_THREADS=1
export NSYS_GRAPH_TRACE=node NSYS_EVENT_TRACE=true A8B_GRAPH_SCOPE=moe
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT=12m
export SOFT_CLIQUE_TRAIN_ITERS=40 SOFT_CLIQUE_PROFILE_START=25 SOFT_CLIQUE_PROFILE_END=28
export SOFT_CLIQUE_TIMING_START=25 SOFT_CLIQUE_TIMING_END=27
export A8B_DEPTH_EXPECTED_SHA=677332c72a1242ccd61334901c60b37c34500f021ecc96a527ffeec09b28966d
git -C "$candidate" diff --quiet HEAD
for CASE_ORDER in ${ATTRIBUTION_CASES:-nep healthy}; do
    export CASE_ORDER SOFT_CLIQUE_TAG="${ATTRIBUTION_TAG:-graph_attribution_${CASE_ORDER}}"
    echo "ATTRIBUTION_BEGIN $(date --iso-8601=seconds) $CASE_ORDER"
    set +e
    bash "$candidate/tests/unit_tests/distributed/nep_param_transfer_soft_clique.sh"
    status=$?
    set -e
    ((status == 0 || status == 1)) || exit "$status"
    python3 - "$base" "$SLURM_JOB_ID" "$SOFT_CLIQUE_TAG" "$CASE_ORDER" <<'PY'
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / 'scripts/nonuniform'))
from training_scripts_nep_matrix import STATUS_PATTERN, TIMING_PATTERN

job, tag, case = sys.argv[2:]
root = Path('/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline')
run = root / job / tag / 'nsys/ep16/a8b_120b_latentmoe_1t' / case / job
log = (run / 'driver.log').read_text()
statuses = [tuple(map(int, m.groups())) for m in STATUS_PATTERN.finditer(log)][-40:]
assert statuses == [(i, 40, 0, 0) for i in range(1, 41)], statuses
assert re.search(r'moe_router_force_uniform_routing\s+\.+\s+True', log)
assert 'token drop during MoE token dispatch' not in log
world = 32 if case == 'healthy' else 28
for folder, suffix in [('cupti_profile', 'json'), ('nsys_profile', 'nsys-rep'), ('nsys_profile', 'sqlite')]:
    assert len(list((run / folder).glob(f'rank-*.{suffix}'))) == world
result = dict(case=case, run_dir=str(run), dual_profile=True,
              timings={int(i):float(ms) for i,ms in TIMING_PATTERN.findall(log)})
(run / 'baseline_uniform.json').write_text(json.dumps(result, indent=2) + '\n')
print('UNIFORM_CASE_VALIDATED', case)
PY
done
