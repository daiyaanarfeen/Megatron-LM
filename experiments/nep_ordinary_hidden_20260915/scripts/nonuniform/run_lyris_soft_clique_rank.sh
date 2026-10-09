#!/bin/bash
set -euo pipefail
read -r -a topology <<< "${NONUNIFORM_EP_TOPOLOGY}"
[[ ${#topology[@]} == 2 ]]
axis=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
full_ranks=$((topology[0] * axis))
[[ "${RUN_WORLD_SIZE}" == "$(((topology[0] + topology[1]) * axis))" ]]
((RANK >= 0 && RANK < RUN_WORLD_SIZE))
# Match the native contiguous Slurm distribution, including partially used nodes.
run_nodes=$(((RUN_WORLD_SIZE + 3) / 4))
base=$((RUN_WORLD_SIZE / run_nodes))
extra=$((RUN_WORLD_SIZE % run_nodes))
offset=0
for ((host_index=0; host_index<run_nodes; host_index++)); do
    count=$((base + (host_index < extra)))
    if ((RANK < offset + count)); then
        local_gpu=$((RANK - offset))
        break
    fi
    offset=$((offset + count))
done
read -r -a hosts <<< "${SOFT_CLIQUE_HOSTS}"
[[ "$(hostname -s)" == "${hosts[host_index]}" ]]
[[ "${SLURM_LOCALID}" == "${local_gpu}" ]]
replica_index=$((RANK >= full_ranks))
export NCCL_MNNVL_CLIQUE_ID=$((1000 + replica_index))
if [[ "${SOFT_CLIQUE_DRY_RUN:-0}" == 1 ]]; then
    echo "rank=${RANK} host_index=${host_index} gpu=${local_gpu} clique=${NCCL_MNNVL_CLIQUE_ID}"
    exit 0
fi
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,P2P,ENV
export NCCL_DEBUG_FILE="${SOFT_CLIQUE_ROOT}/transport/nccl.%h.%p.log"
mkdir -p "${SOFT_CLIQUE_ROOT}/placement/${NONUNIFORM_MODE}"
printf 'rank=%s host=%s gpu=%s replica=%s clique=%s\n' "${RANK}" "$(hostname -s)" "${CUDA_VISIBLE_DEVICES}" "${replica_index}" "${NCCL_MNNVL_CLIQUE_ID}" | tee "${SOFT_CLIQUE_ROOT}/placement/${NONUNIFORM_MODE}/rank-${RANK}.txt"
nvidia-smi -q -i "${CUDA_VISIBLE_DEVICES}" > "${SOFT_CLIQUE_ROOT}/placement/${NONUNIFORM_MODE}/gpu-${RANK}.txt"
exec python "$(dirname "${BASH_SOURCE[0]}")/run_nep_ordinary_hidden_rank.py"
