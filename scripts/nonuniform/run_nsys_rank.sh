#!/bin/bash
set -euo pipefail
: "${NSYS_OUTPUT_DIR:?}"
: "${RANK:?}"
mkdir -p "${NSYS_OUTPUT_DIR}/tmp-rank-${RANK}"
export NSYS_TMPDIR="${NSYS_OUTPUT_DIR}/tmp-rank-${RANK}"
# Python multiprocessing sockets need a short, node-local temporary path.
export TMPDIR=/tmp
prefix="${NSYS_OUTPUT_DIR}/rank-${RANK}"
nsys --version > "${prefix}.version.txt"
python -c 'import nvtx; nvtx.push_range("preflight", domain="nep_capture"); nvtx.pop_range(domain="nep_capture")'
TMPDIR="${NSYS_TMPDIR}" nsys profile --trace="${NSYS_CUDA_TRACE:-cuda-sw},nvtx" --nvtx-domain-include=nep_capture \
    --sample=none --cpuctxsw=none --cuda-graph-trace="${NSYS_GRAPH_TRACE:-graph:host-only}" --cuda-event-trace="${NSYS_EVENT_TRACE:-false}" \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true --output="${prefix}" \
    env TMPDIR=/tmp bash "$(dirname "${BASH_SOURCE[0]}")/run_lyris_a3b_direct_rank.sh"
nsys export --type=sqlite --force-overwrite=true --output="${prefix}.sqlite" \
    "${prefix}.nsys-rep" > "${prefix}.export.log" 2>&1
