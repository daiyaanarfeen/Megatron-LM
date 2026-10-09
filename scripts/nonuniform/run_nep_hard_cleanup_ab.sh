#!/bin/bash
#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill,gb300-backfill
#SBATCH --nodes=8
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:35:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --job-name=coreai_comparch_sysarch-nep.hard-cleanup
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
ulimit -c 0
base=/home/darfeen/Megatron-LM
reference="$base/.worktrees/nep-graph-launch-order-20260921"
candidate="$reference"
export NEP_LAUNCH_ORDER_IMPLICIT=${NEP_LAUNCH_ORDER_IMPLICIT:-0}
export PARAM_TRANSFER_CODE_REPO="$candidate" NEP_SYNC_REFERENCE="$candidate"
unset NEP_SYNC_SHADOW SYNC_WORKLOAD_VARIANT NEP_TAIL_MODE
export NEP_SYNC_ENTRYPOINT="$base/scripts/nonuniform/pretrain_nep_hard_cleanup_probe.py"
export SOFT_CLIQUE_MBS=1 SOFT_CLIQUE_PROFILE_SWEEP=1 PROFILE_ACTIVITY_MODES=nsys
export A8B_GRAPH_EXPECTED_MBS=1 A8B_GRAPH_EXPECTED_GA=2
export CUPTI_EVENTS_ONLY=1 CUPTI_COLLECTION_DISABLED=0 CUDA_EVENT_POOL_SIZE=32768 OMP_NUM_THREADS=1
export NSYS_GRAPH_TRACE=node NSYS_EVENT_TRACE=true A8B_GRAPH_SCOPE=moe
export GROUP=ep16 WORKLOAD_FILTER=a8b_120b_latentmoe_1t CASE_TIMEOUT=8m
export SOFT_CLIQUE_TRAIN_ITERS=35 SOFT_CLIQUE_PROFILE_START=20 SOFT_CLIQUE_PROFILE_END=28
export SOFT_CLIQUE_TIMING_START=20 SOFT_CLIQUE_TIMING_END=27
for variant in ${HARD_CASES:-healthy reference cleanup parallel parallel_margin healthy_margin}; do
    candidate="$reference"
    export CASE_ORDER=nep NEP_TEST_BWD_SM_MARGIN=16
    case "$variant" in
        ablation_*)
            candidate="${ABLATION_CODE_REPO:-$base/.worktrees/nep-placement-ablation-20260928}"
            export NEP_TEST_BWD_SM_MARGIN=32
            export NEP_ABLATION_PLACEMENT=current NEP_ABLATION_PARAM_ORDER=waves NEP_ABLATION_GRAD_ORDER=waves
            export NEP_ABLATION_PARAM_RELEASE=early NEP_ABLATION_GRAD_RELEASE=early
            case "$variant" in
                ablation_healthy*) export CASE_ORDER=healthy ;;
                ablation_current_ascending) export NEP_ABLATION_PARAM_ORDER=ascending ;;
                ablation_spread_waves) export NEP_ABLATION_PLACEMENT=spread ;;
                ablation_spread_ascending) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_PARAM_ORDER=ascending ;;
                ablation_grad_ascending) export NEP_ABLATION_GRAD_ORDER=ascending ;;
                ablation_spread_grad_ascending) export NEP_ABLATION_PLACEMENT=spread NEP_ABLATION_GRAD_ORDER=ascending ;;
                ablation_late_params) export NEP_ABLATION_PARAM_RELEASE=consumer ;;
                ablation_late_grads) export NEP_ABLATION_GRAD_RELEASE=finish ;;
                ablation_current_waves|ablation_current_repeat) ;;
                *) exit 2 ;;
            esac
            ;;
        healthy) export CASE_ORDER=healthy ;;
        healthy_margin) export CASE_ORDER=healthy NEP_TEST_BWD_SM_MARGIN=32 ;;
        reference) ;;
        cleanup) candidate="$base/.worktrees/nep-buffer-cleanup-20260921" ;;
        cleanup_margin) candidate="$base/.worktrees/nep-buffer-cleanup-20260921"; export NEP_TEST_BWD_SM_MARGIN=32 ;;
        ag_overlap_margin) candidate="$base/.worktrees/nep-ag-red-overlap-20260922"; export NEP_TEST_BWD_SM_MARGIN=32 ;;
        param_waves_margin) candidate="$base/.worktrees/nep-param-waves-20260922"; export NEP_TEST_BWD_SM_MARGIN=32 ;;
        parallel) candidate="$base/.worktrees/nep-param-owner-parallel-20260921" ;;
        parallel_margin) candidate="$base/.worktrees/nep-param-owner-parallel-20260921"; export NEP_TEST_BWD_SM_MARGIN=32 ;;
        *) exit 2 ;;
    esac
    git -C "$candidate" diff --quiet HEAD
    export PARAM_TRANSFER_CODE_REPO="$candidate" NEP_SYNC_REFERENCE="$candidate"
    export A8B_DEPTH_EXPECTED_SHA=$(sha256sum "$candidate/megatron/core/distributed/nonuniform_ep.py" | awk '{print $1}')
    export SOFT_CLIQUE_TAG="cleanup_ab_${variant}"
    echo "ATTRIBUTION_BEGIN $(date --iso-8601=seconds) $CASE_ORDER"
    set +e
    bash "$base/experiments/nep_cleanup_20260921/nep_param_transfer_soft_clique.sh"
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
statuses = [tuple(map(int, m.groups())) for m in STATUS_PATTERN.finditer(log)][-35:]
assert statuses == [(i, 35, 0, 0) for i in range(1, 36)], statuses
assert re.search(r'moe_router_force_uniform_routing\s+\.+\s+True', log)
assert 'token drop during MoE token dispatch' not in log
world = 32 if case == 'healthy' else 28
expected_gbs = world
assert re.search(rf'global batch size:\s+{expected_gbs}\s*\|', log)
for path in (run / 'launch_manifest').glob('rank-*.json'):
    manifest = json.loads(path.read_text())
    assert manifest['replica_mbs'] == '1 1' and manifest['replica_ga'] == '2 2'
    options = manifest['options']
    assert options[options.index('--micro-batch-size') + 1] == '1'
    assert options[options.index('--global-batch-size') + 1] == str(expected_gbs)
assert len(list((run / 'launch_manifest').glob('rank-*.json'))) == world
for folder, suffix in [('cupti_profile', 'json'), ('nsys_profile', 'nsys-rep'), ('nsys_profile', 'sqlite')]:
    assert len(list((run / folder).glob(f'rank-*.{suffix}'))) == world
result = dict(case=case, run_dir=str(run), dual_profile=True,
              timings={int(i):float(ms) for i,ms in TIMING_PATTERN.findall(log)})
(run / 'baseline_uniform.json').write_text(json.dumps(result, indent=2) + '\n')
print('UNIFORM_CASE_VALIDATED', case)
PY
done
