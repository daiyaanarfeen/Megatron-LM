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
#SBATCH --job-name=coreai_comparch_sysarch-nep.graph-launch-order
#SBATCH --output=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/%x-%j.err
set -euo pipefail
base=/home/darfeen/Megatron-LM
export GRAPH_CODE_REPO="$base/.worktrees/nep-graph-launch-order-20260921"
for variant in healthy:1 nep:1 nep:0 healthy:0; do
    export ATTRIBUTION_CASES=${variant%:*} NEP_LAUNCH_ORDER_IMPLICIT=${variant#*:}
    export ATTRIBUTION_TAG="launch_order${NEP_LAUNCH_ORDER_IMPLICIT}_${ATTRIBUTION_CASES}"
    echo "LAUNCH_ORDER_BEGIN $variant $(date --iso-8601=seconds)"
    bash "$base/scripts/nonuniform/run_a8b_graph_attribution.sh"
    echo "LAUNCH_ORDER_DONE $variant $(date --iso-8601=seconds)"
done
