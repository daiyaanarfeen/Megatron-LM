#!/bin/bash

# EP8/EP6 nondivisible-expert correctness gate with Flex/HybridEP and native Adam.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=4
#SBATCH --segment=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-6-flex-opt

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_ep8_ep6_flex_regular_optimizer}"
RUNNER="${REPO_DIR}/examples/training_scripts/nonuniform_ep_approach_a_smoke.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-8m}"
NAME="ep8_ep6_n16_flex_hybridep_adam_${SLURM_JOB_ID}"
RUN_DIR="${ROOT_DIR}/${NAME}"
CHECKSUM_DIR="${RUN_DIR}/checksums"
DRIVER_LOG="${RUN_DIR}/driver_${SLURM_JOB_ID}.log"

mkdir -p "${RUN_DIR}" "${CHECKSUM_DIR}" "${REPO_DIR}/slurm_runs/lyris"
exec > >(tee -a "${DRIVER_LOG}") 2>&1

container_args=(
    --container-image="${IMAGE}"
    --container-mounts="${REPO_DIR}:${REPO_DIR}"
    --container-workdir="${REPO_DIR}"
    --no-container-mount-home
)
if [[ -n "${CONTAINER_NAME}" ]]; then
    container_args+=(--container-name="${CONTAINER_NAME}")
fi

mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if ((${#allocated_nodes[@]} != 4)); then
    echo "Expected four allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
RUN_NODELIST=$(IFS=,; echo "${allocated_nodes[*]}")
MASTER_ADDR="${allocated_nodes[0]}"

# Warm the named Enroot cache on every selected node before preflight/training.
srun --nodes=4 --nodelist="${RUN_NODELIST}" --ntasks=4 --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" bash -lc 'python -c "import torch; print(torch.__version__)"'

# Required style checks plus focused logical-placement and Flex metadata tests.
srun --nodes=1 --nodelist="${MASTER_ADDR}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${REPO_DIR}'
python -m black --required-version 26 \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m isort \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m isort --check-only \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m black --required-version 26 --check \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m py_compile \
    megatron/core/distributed/nonuniform_common.py \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/transformer/moe/moe_layer.py \
    megatron/core/transformer/moe/token_dispatcher.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m pytest -q tests/unit_tests/distributed/test_nonuniform_ep.py \
    -k 'nondivisible_ep8_ep6 or flex_metadata_maps or nonuniform_expert_placement or zero_sm_expert_placement or process_group_expert_placement or ep64_ep48'
"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT=1
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export MEGATRON_NONUNIFORM_EP_DEBUG=1
export MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
export MEGATRON_NONUNIFORM_EP_OVERLAP_DEBUG=0
export MEGATRON_NONUNIFORM_EP_ZERO_SM_RESHARD=0
export MEGATRON_NONUNIFORM_EP_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_BUCKET_READY_GATHER=1
export MEGATRON_NONUNIFORM_EP_DEVICE_ORDERED_EDP=1
export MEGATRON_NONUNIFORM_EP_HOST_EDP_READY_GATE=0
export MEGATRON_NONUNIFORM_EP_SAME_COMM_READY=0
export MEGATRON_NONUNIFORM_EP_DEFER_HOST_LAUNCH=0
export MEGATRON_NONUNIFORM_EP_DEFER_MODEL_EP_FENCE=1
export MEGATRON_NONUNIFORM_EP_A2A_SCATTER_SCHEDULER=0
export MEGATRON_NONUNIFORM_EP_END_ITERATION_SCATTER=1
export MEGATRON_NONUNIFORM_EP_PIPELINE_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_SPLIT_HOST_PHASES=1
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_PHASES=0
export MEGATRON_NONUNIFORM_EP_POST_GRAPH_HOST_PHASES=0
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS=2
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1

extra_args="--moe-flex-dispatcher-backend hybridep \
--moe-router-dtype fp32 \
--moe-permute-fusion \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--nonuniform-log-param-checksum \
--nonuniform-param-checksum-dir ${CHECKSUM_DIR} \
--distributed-timeout-minutes 3"

echo "[ep8-ep6-flex-opt] $(date --iso-8601=seconds) starting ${NAME}"
timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
    srun --nodes=4 --nodelist="${RUN_NODELIST}" \
        --ntasks=14 --ntasks-per-node=4 --kill-on-bad-exit=1 --mpi=none \
        "${container_args[@]}" \
        bash -lc "
export RANK=\"\${SLURM_PROCID}\"
export LOCAL_RANK=0
export TORCH_CUDA_VISIBLE_DEVICES=\"\${SLURM_LOCALID}\"
export REPO_DIR='${REPO_DIR}'
export ROOT_DIR='${ROOT_DIR}'
export NAME='${NAME}'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT=29961
export RUN_DIRECT=1
export LAUNCHER_MODE=direct
export IMAGE_PATH='${IMAGE}'
export NNODES=4
export GPUS_PER_NODE=4
export NPROC_PER_NODE=4
export WORLD_SIZE=14
export TRAIN_ITERS=4
export GLOBAL_BATCH_SIZE=14
export MICRO_BATCH_SIZE=1
export NUM_LAYERS=2
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1024
export NUM_ATTENTION_HEADS=4
export SEQ_LENGTH=128
export NUM_EXPERTS=16
export TENSOR_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export EXPERT_TENSOR_PARALLEL_SIZE=1
export NONUNIFORM_MODE=ep
export NONUNIFORM_EP_DDP_APPROACH=nccl
export NONUNIFORM_EP_TOPOLOGY='8 6'
export MOE_TOKEN_DISPATCHER_TYPE=flex
export NONUNIFORM_SKIP_OPTIMIZER_STEP=0
export ENABLE_PYTORCH_PROFILER=1
export PROFILE_STEP_START=1
export PROFILE_STEP_END=3
export PROFILE_RANKS='0 8'
export EXTRA_MEGATRON_ARGS='${extra_args}'
bash '${RUNNER}'
"
echo "[ep8-ep6-flex-opt] $(date --iso-8601=seconds) training completed"

python3 - "${DRIVER_LOG}" "${CHECKSUM_DIR}" <<'PY'
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
checksum_dir = Path(sys.argv[2])
text = log_path.read_text()

placement_pattern = re.compile(
    r"initialize_nonuniform_ep_process_groups exit rank=(\d+) "
    r"local_ep_size=(\d+) ep_rank=(\d+) local_expert_indices=\[([^]]*)\]"
)
placements = {}
for match in placement_pattern.finditer(text):
    rank, ep_size, ep_rank = map(int, match.group(1, 2, 3))
    experts = [int(value.strip()) for value in match.group(4).split(',') if value.strip()]
    placements[(ep_size, ep_rank)] = (rank, experts)

expected = {
    (8, 0): [0, 1],
    (8, 1): [3, 4],
    (8, 2): [6, 7],
    (8, 3): [9, 10],
    (8, 4): [12, 13],
    (8, 5): [14, 15],
    (8, 6): [2, 5],
    (8, 7): [8, 11],
    (6, 0): [0, 1, 2],
    (6, 1): [3, 4, 5],
    (6, 2): [6, 7, 8],
    (6, 3): [9, 10, 11],
    (6, 4): [12, 13],
    (6, 5): [14, 15],
}
if {key: value[1] for key, value in placements.items()} != expected:
    raise RuntimeError(f"Unexpected logical expert placement: {placements}")

if not re.search(r"moe_token_dispatcher_type\s+\.+\s+flex", text):
    raise RuntimeError("Run did not report the Flex token dispatcher")
if not re.search(r"moe_flex_dispatcher_backend\s+\.+\s+hybridep", text):
    raise RuntimeError("Run did not report the HybridEP Flex backend")
if not re.search(r"use_distributed_optimizer\s+\.+\s+False", text):
    raise RuntimeError("Run did not report the regular non-distributed optimizer")

number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
fields = (
    "weighted_sum",
    "weighted_abs",
    "weighted_sq",
    "weighted_numel",
    "dense_weighted_sum",
    "dense_weighted_abs",
    "dense_weighted_sq",
    "dense_weighted_numel",
    "dense_numel",
    "expert_weighted_sum",
    "expert_weighted_abs",
    "expert_weighted_sq",
    "expert_weighted_numel",
    "expert_numel",
    "local_expert_numel",
)
pattern = re.compile(
    r"\[nonuniform-param-checksum\] iteration=(\S+) rank=(\d+) "
    + " ".join(rf"{field}=({number})" for field in fields)
)
records = {}
for path in sorted(checksum_dir.glob("rank_*.log")):
    for match in pattern.finditer(path.read_text()):
        iteration = int(match.group(1))
        rank = int(match.group(2))
        values = tuple(float(value) for value in match.groups()[2:])
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite parameter checksum at iteration={iteration} rank={rank}")
        records.setdefault(iteration, {})[rank] = values

if len(records) < 3:
    raise RuntimeError(f"Expected at least three post-step checksum iterations, got {records.keys()}")
for iteration, rows in records.items():
    if len(rows) != 14:
        raise RuntimeError(f"Iteration {iteration} has {len(rows)} checksum ranks, expected 14")

rank_for = {key: value[0] for key, value in placements.items()}
full_reference_rank = rank_for[(8, 0)]
reduced_reference_rank = rank_for[(6, 0)]
expert_fields = range(9, 14)
exact_fields = {7, 8, 12, 13, 14}


def assert_close(label, lhs, rhs, exact=False):
    if exact:
        if lhs != rhs:
            raise RuntimeError(f"{label}: exact mismatch {lhs} != {rhs}")
    elif not math.isclose(lhs, rhs, rel_tol=2e-6, abs_tol=1e-7):
        raise RuntimeError(f"{label}: mismatch {lhs} != {rhs}")


first_iteration, last_iteration = min(records), max(records)
for rank in range(14):
    first = records[first_iteration][rank]
    last = records[last_iteration][rank]
    if all(
        math.isclose(first[index], last[index], rel_tol=1e-12, abs_tol=1e-12)
        for index in (0, 1, 2)
    ):
        raise RuntimeError(f"Regular optimizer did not change rank {rank}'s parameter fingerprint")
if all(
    math.isclose(
        records[first_iteration][full_reference_rank][index],
        records[last_iteration][full_reference_rank][index],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    for index in (9, 10, 11)
):
    raise RuntimeError("Regular optimizer did not change the logical expert fingerprint")

for iteration, rows in sorted(records.items()):
    # The logger all-reduces expert fingerprints over each complete EP replica.
    # Therefore compare one replica-level fingerprint from EP8 directly with one
    # from EP6; summing rank records would count the same fingerprint repeatedly.
    full_expert_reference = rows[full_reference_rank]
    reduced_expert_reference = rows[reduced_reference_rank]
    for field in expert_fields:
        assert_close(
            f"iteration={iteration} cross-replica {fields[field]}",
            full_expert_reference[field],
            reduced_expert_reference[field],
            exact=field in exact_fields,
        )

    for ep_size in (8, 6):
        reference_rank = full_reference_rank if ep_size == 8 else reduced_reference_rank
        for ep_rank in range(ep_size):
            rank = rank_for[(ep_size, ep_rank)]
            for field in expert_fields:
                assert_close(
                    f"iteration={iteration} EP{ep_size} rank={rank} {fields[field]}",
                    rows[reference_rank][field],
                    rows[rank][field],
                    exact=field in exact_fields,
                )

    dense_reference = rows[full_reference_rank][4:9]
    for rank, row in rows.items():
        for offset, (reference, value) in enumerate(zip(dense_reference, row[4:9]), start=4):
            assert_close(
                f"iteration={iteration} rank={rank} {fields[offset]}",
                reference,
                value,
                exact=offset in exact_fields,
            )

    # local_expert_numel is captured before the replica-wide checksum all-reduce,
    # so it proves that virtual Flex slots did not create parameters or optimizer state.
    full_two_expert_numel = rows[full_reference_rank][14]
    for ep_rank in range(8):
        assert_close(
            f"iteration={iteration} EP8 rank={ep_rank} local_expert_numel",
            rows[rank_for[(8, ep_rank)]][14],
            full_two_expert_numel,
            exact=True,
        )
    for ep_rank in range(4):
        assert_close(
            f"iteration={iteration} EP6 rank={ep_rank} has exactly three logical experts",
            rows[rank_for[(6, ep_rank)]][14] * 2,
            full_two_expert_numel * 3,
            exact=True,
        )
    for ep_rank in (4, 5):
        assert_close(
            f"iteration={iteration} EP6 rank={ep_rank} has no dummy parameters",
            rows[rank_for[(6, ep_rank)]][14],
            full_two_expert_numel,
            exact=True,
        )

    for ep_size, reference_rank in ((8, full_reference_rank), (6, reduced_reference_rank)):
        local_total = sum(rows[rank_for[(ep_size, ep_rank)]][14] for ep_rank in range(ep_size))
        assert_close(
            f"iteration={iteration} EP{ep_size} local/replica expert numel",
            local_total,
            rows[reference_rank][13],
            exact=True,
        )

print(
    "[ep8-ep6-flex-opt] PASS: balanced 16-logical-expert placement, zero dummy params, "
    "Flex/HybridEP dispatch, native optimizer updates, and cross-replica parameter equality"
)
PY
