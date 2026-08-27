#!/bin/bash

# Set HybridEP's NVLink-domain participant count per nonuniform replica, then
# delegate batch-size calculation and launch to the validated direct runner.

set -euo pipefail

: "${RANK:?RANK must be set}"
: "${NONUNIFORM_MODE:?NONUNIFORM_MODE must be set}"
: "${NONUNIFORM_EP_TOPOLOGY:?NONUNIFORM_EP_TOPOLOGY must be set}"
: "${TENSOR_MODEL_PARALLEL_SIZE:?TENSOR_MODEL_PARALLEL_SIZE must be set}"
: "${CONTEXT_PARALLEL_SIZE:?CONTEXT_PARALLEL_SIZE must be set}"
: "${BASE_RANK_LAUNCHER:?BASE_RANK_LAUNCHER must be set}"

if [[ "${NONUNIFORM_MODE}" == "ep" ]]; then
    read -r -a topology <<< "${NONUNIFORM_EP_TOPOLOGY}"
    tp_cp=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
    rank_offset=0
    replica_domain_size=0
    for replica_tp_cp in "${topology[@]}"; do
        replica_ranks=$((replica_tp_cp * tp_cp))
        if ((RANK >= rank_offset && RANK < rank_offset + replica_ranks)); then
            replica_domain_size=${replica_ranks}
            break
        fi
        rank_offset=$((rank_offset + replica_ranks))
    done
    if ((replica_domain_size == 0)); then
        echo "Rank ${RANK} does not belong to topology ${NONUNIFORM_EP_TOPOLOGY}" >&2
        exit 2
    fi
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${replica_domain_size}"
else
    if [[ -n "${SOURCE_HYBRID_EP_DOMAIN_SIZE:-}" ]]; then
        export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${SOURCE_HYBRID_EP_DOMAIN_SIZE}"
    else
        unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN || true
    fi
fi
exec bash "${BASE_RANK_LAUNCHER}"
