#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port
LOGROOT=$WORKDIR/ep32_mbs4_distopt30_20260527_130433
STAMP=$LOGROOT/submitted_jobid.txt

mkdir -p "$LOGROOT"

if [[ ! -s "$STAMP" ]]; then
  sbatch \
    --chdir="$WORKDIR" \
    --job-name=ep32_mbs4_distopt30 \
    --time=00:15:00 \
    --export=ALL,RUN_CASES=current_cmd1,TRAIN_ITERS=30,MICRO_BATCH_SIZE=4,USE_DISTRIBUTED_OPTIMIZER=1,OVERLAP_PARAM_GATHER=1,MOE_EXPERT_CAPACITY_FACTOR=1.05,MOE_PAD_EXPERT_INPUT_TO_CAPACITY=1,PROFILE=0,BASE_LOG_ROOT="$LOGROOT" \
    tools/hsg_ep32_uniform_baseline_sweep.sbatch > "$STAMP"
fi

exec rsync "$@"
