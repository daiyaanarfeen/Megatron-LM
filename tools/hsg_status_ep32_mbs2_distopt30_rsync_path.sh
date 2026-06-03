#!/usr/bin/env bash
set -euo pipefail

WORKDIR=/lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port
LOGROOT=$WORKDIR/ep32_mbs2_distopt30_20260527_113723
STAMP=$LOGROOT/submitted_jobid.txt
STATUS=$LOGROOT/status.txt

mkdir -p "$LOGROOT"
if [[ -s "$STAMP" ]]; then
  JOB_ID=$(awk '{print $NF}' "$STAMP" | tail -n1)
  {
    date
    squeue -j "$JOB_ID" -o "%.18i %.9P %.32j %.8u %.2t %.10M %.10L %.20S %.30R" || true
    sacct -j "$JOB_ID" --format=JobID,JobName%32,Partition,State,Elapsed,Start,End,ExitCode -P || true
  } > "$STATUS" 2>&1
else
  {
    date
    echo "No submitted_jobid.txt yet"
  } > "$STATUS"
fi

exec rsync "$@"
