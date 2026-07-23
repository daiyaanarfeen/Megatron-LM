#!/bin/bash

# Resolve replica-local batch settings for one direct SLURM GPU rank.

set -euo pipefail

: "${RANK:?RANK must be set}"
: "${WORLD_SIZE:?WORLD_SIZE must be set}"
: "${NONUNIFORM_EP_TOPOLOGY:?NONUNIFORM_EP_TOPOLOGY must be set}"
: "${TENSOR_MODEL_PARALLEL_SIZE:?TENSOR_MODEL_PARALLEL_SIZE must be set}"
: "${CONTEXT_PARALLEL_SIZE:?CONTEXT_PARALLEL_SIZE must be set}"
: "${MICRO_BATCH_SIZE:?MICRO_BATCH_SIZE must be set}"
: "${GLOBAL_BATCH_SIZE:?GLOBAL_BATCH_SIZE must be set}"
: "${REPO_DIR:?REPO_DIR must be set}"
: "${options:?options must be set}"

read -r -a topology <<< "${NONUNIFORM_EP_TOPOLOGY}"
read -r -a replica_mbs <<< "${REPLICA_MICRO_BATCH_SIZES:-}"
read -r -a replica_num_microbatches <<< "${REPLICA_NUM_MICROBATCHES:-}"

if ((${#topology[@]} == 0)); then
    echo "NONUNIFORM_EP_TOPOLOGY is empty" >&2
    exit 2
fi
if ((${#replica_mbs[@]} != 0 || ${#replica_num_microbatches[@]} != 0)); then
    if ((${#replica_mbs[@]} != ${#topology[@]})); then
        echo "REPLICA_MICRO_BATCH_SIZES must match topology length" >&2
        exit 2
    fi
    if ((${#replica_num_microbatches[@]} != ${#topology[@]})); then
        echo "REPLICA_NUM_MICROBATCHES must match topology length" >&2
        exit 2
    fi
fi

tp_cp=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
dp_size=0
expected_world_size=0
replica_index=-1
rank_offset=0
for index in "${!topology[@]}"; do
    replica_dp_size=${topology[index]}
    replica_world_size=$((replica_dp_size * tp_cp))
    dp_size=$((dp_size + replica_dp_size))
    expected_world_size=$((expected_world_size + replica_world_size))
    if ((RANK >= rank_offset && RANK < rank_offset + replica_world_size)); then
        replica_index=${index}
    fi
    rank_offset=$((rank_offset + replica_world_size))
done

if ((WORLD_SIZE != expected_world_size)); then
    echo "WORLD_SIZE=${WORLD_SIZE} does not match topology world size ${expected_world_size}" >&2
    exit 2
fi
if ((replica_index < 0)); then
    echo "Rank ${RANK} does not belong to topology ${NONUNIFORM_EP_TOPOLOGY}" >&2
    exit 2
fi

if ((${#replica_mbs[@]} == 0)); then
    local_mbs=${MICRO_BATCH_SIZE}
    denominator=$((MICRO_BATCH_SIZE * dp_size))
    if ((GLOBAL_BATCH_SIZE % denominator != 0)); then
        echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} is not divisible by MBS*DP=${denominator}" >&2
        exit 2
    fi
    local_num_microbatches=$((GLOBAL_BATCH_SIZE / denominator))
else
    local_mbs=${replica_mbs[replica_index]}
    local_num_microbatches=${replica_num_microbatches[replica_index]}
fi

if ((local_mbs < 1 || local_num_microbatches < 1)); then
    echo "Replica-local MBS and microbatch count must be positive" >&2
    exit 2
fi

# Megatrons calculator is rank-local. This surrogate GBS produces the desired
# local accumulation count; --calculate-per-token-loss performs the real global
# sample/token weighting after gradient synchronization.
local_calculator_gbs=$((local_mbs * dp_size * local_num_microbatches))

echo "[lyris-a3b-direct] rank=${RANK} replica=${replica_index} mbs=${local_mbs} num_microbatches=${local_num_microbatches} calculator_gbs=${local_calculator_gbs} true_gbs=${TRUE_GLOBAL_BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"

if [[ "${DIRECT_RANK_DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

cd "${REPO_DIR}"
exec python -u pretrain_hybrid.py ${options} \
    --micro-batch-size "${local_mbs}" \
    --global-batch-size "${local_calculator_gbs}"
