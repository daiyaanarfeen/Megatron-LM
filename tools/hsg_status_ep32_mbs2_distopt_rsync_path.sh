#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port
LOGROOT=$WORKDIR/ep32_mbs2_distopt_20260527_110843
JOB_ID=2970782

mkdir -p "$LOGROOT"
{
  date
  squeue -j "$JOB_ID" -o "%.18i %.9P %.32j %.8u %.2t %.10M %.10L %.20S %.30R" || true
  sacct -j "$JOB_ID" --format=JobID,JobName%32,Partition,State,Elapsed,Start,End,ExitCode -P || true
} > "$LOGROOT/status_${JOB_ID}.txt" 2>&1

exec rsync "$@"
