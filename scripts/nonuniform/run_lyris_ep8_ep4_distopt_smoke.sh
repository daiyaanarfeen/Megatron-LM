#!/bin/bash

# Bounded EP8/EP4 correctness and trace gate for NEP owner-layout distributed optimizer.

#SBATCH --account=coreai_comparch_sysarch
#SBATCH --partition=gb200-backfill
#SBATCH --nodes=3
#SBATCH --segment=3
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/darfeen/Megatron-LM
#SBATCH --output=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.out
#SBATCH --error=/home/darfeen/Megatron-LM/slurm_runs/lyris/%x-%j.err
#SBATCH --job-name=coreai_comparch_sysarch-nep.ep8-4-distopt

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/darfeen/Megatron-LM}"
ROOT_DIR="${ROOT_DIR:-${REPO_DIR}/slurm_runs/lyris_ep8_ep4_distopt_smoke}"
RUNNER="${REPO_DIR}/examples/training_scripts/nonuniform_ep_approach_a_smoke.sh"
IMAGE="${IMAGE:-nvcr.io#nvidia/nemo:26.06}"
CONTAINER_NAME="${CONTAINER_NAME:-nep_nemo_26_06}"
CASE_TIMEOUT="${CASE_TIMEOUT:-8m}"
OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-0}"
RUN_NNODES="${RUN_NNODES:-3}"
FULL_EP_SIZE="${FULL_EP_SIZE:-8}"
REDUCED_EP_SIZE="${REDUCED_EP_SIZE:-4}"
MODEL_NUM_EXPERTS="${MODEL_NUM_EXPERTS:-16}"
TOKEN_DISPATCHER="${TOKEN_DISPATCHER:-alltoall}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
EXPERT_TENSOR_PARALLEL_SIZE="${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
NUM_LAYERS="${NUM_LAYERS:-2}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-1024}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-4}"
SEQ_LENGTH="${SEQ_LENGTH:-128}"
parallel_unit=$((TENSOR_MODEL_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE))
if ((FULL_EP_SIZE * EXPERT_TENSOR_PARALLEL_SIZE % parallel_unit != 0)) || \
    ((REDUCED_EP_SIZE * EXPERT_TENSOR_PARALLEL_SIZE % parallel_unit != 0)); then
    echo "EP sizes must map to whole TP*CP units" >&2
    exit 2
fi
FULL_TOPOLOGY_UNIT=$((FULL_EP_SIZE * EXPERT_TENSOR_PARALLEL_SIZE / parallel_unit))
REDUCED_TOPOLOGY_UNIT=$((REDUCED_EP_SIZE * EXPERT_TENSOR_PARALLEL_SIZE / parallel_unit))
RUN_WORLD_SIZE=$(((FULL_EP_SIZE + REDUCED_EP_SIZE) * EXPERT_TENSOR_PARALLEL_SIZE))
PROFILE_RANKS=$(seq -s ' ' 0 $((RUN_WORLD_SIZE - 1)))
NAME="ep${FULL_EP_SIZE}_ep${REDUCED_EP_SIZE}_${TOKEN_DISPATCHER}_distopt_opg${OVERLAP_PARAM_GATHER}_${SLURM_JOB_ID}"
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
if ((${#allocated_nodes[@]} != RUN_NNODES)); then
    echo "Expected ${RUN_NNODES} allocated nodes, got ${#allocated_nodes[@]}" >&2
    exit 2
fi
RUN_NODELIST=$(IFS=,; echo "${allocated_nodes[*]}")
MASTER_ADDR="${allocated_nodes[0]}"

# Warm the named Enroot cache on every node used by the test.
srun --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
    --ntasks="${RUN_NNODES}" --ntasks-per-node=1 --mpi=none \
    "${container_args[@]}" bash -lc 'python -c "import torch; print(torch.__version__)"'

# Verify formatting, then run focused distributed unit tests before training.
srun --nodes=1 --nodelist="${MASTER_ADDR}" --ntasks=1 --mpi=none \
    "${container_args[@]}" bash -lc "
set -euo pipefail
cd '${REPO_DIR}'
python -m isort --check-only \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m black --required-version 26 --check \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m ruff check \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
python -m py_compile \
    megatron/core/distributed/nonuniform_ep.py \
    megatron/core/optimizer/__init__.py \
    examples/nonuniform/pretrain_gpt_nonuniform.py \
    pretrain_hybrid.py \
    tests/unit_tests/distributed/test_nonuniform_ep.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \
    --nproc-per-node=4 -m pytest -q \
    tests/unit_tests/distributed/test_nonuniform_ep.py -k 'nep_distopt'
"

export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_LAUNCH_ORDER_IMPLICIT="${NCCL_LAUNCH_ORDER_IMPLICIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_NVLS_ENABLE=0
export MEGATRON_NONUNIFORM_EP_DEBUG=1
export MEGATRON_NONUNIFORM_EP_DEBUG_RANKS="${PROFILE_RANKS}"
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
export MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS="${MEGATRON_NONUNIFORM_EP_NCCL_EXPERT_BUCKET_GROUPS:-2}"
export MEGATRON_NONUNIFORM_EP_NCCL_GATHER_BUCKETS_PER_EDP=1
export MEGATRON_NONUNIFORM_EP_NCCL_TARGET_CHUNKS=1
export MEGATRON_NONUNIFORM_EP_NCCL_SCATTER_CHUNKS=1

extra_args="--use-distributed-optimizer \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--nonuniform-log-param-checksum \
--nonuniform-param-checksum-dir ${CHECKSUM_DIR} \
--distributed-timeout-minutes 3"
if [[ "${OVERLAP_PARAM_GATHER}" == "1" ]]; then
    extra_args+=" --overlap-param-gather"
fi
if [[ "${TOKEN_DISPATCHER}" == "flex" ]]; then
    extra_args+=" --moe-flex-dispatcher-backend hybridep --moe-router-dtype fp32 --moe-permute-fusion"
fi

echo "[nep-distopt-smoke] $(date --iso-8601=seconds) starting ${NAME}"
timeout --foreground --signal=TERM --kill-after=45s "${CASE_TIMEOUT}" \
    srun --nodes="${RUN_NNODES}" --nodelist="${RUN_NODELIST}" \
        --ntasks="${RUN_WORLD_SIZE}" --ntasks-per-node=4 --kill-on-bad-exit=1 --mpi=none \
        "${container_args[@]}" \
        bash -lc "
export RANK=\"\${SLURM_PROCID}\"
export LOCAL_RANK=0
export TORCH_CUDA_VISIBLE_DEVICES=\"\${SLURM_LOCALID}\"
export REPO_DIR='${REPO_DIR}'
export ROOT_DIR='${RUN_DIR}'
export NAME='training'
export MASTER_ADDR='${MASTER_ADDR}'
export MASTER_PORT=29971
export RUN_DIRECT=1
export LAUNCHER_MODE=direct
export IMAGE_PATH='${IMAGE}'
export NNODES='${RUN_NNODES}'
export GPUS_PER_NODE=4
export NPROC_PER_NODE=4
export WORLD_SIZE='${RUN_WORLD_SIZE}'
export TRAIN_ITERS=4
export GLOBAL_BATCH_SIZE='${GLOBAL_BATCH_SIZE:-$((RUN_WORLD_SIZE / TENSOR_MODEL_PARALLEL_SIZE))}'
export MICRO_BATCH_SIZE=1
export NUM_LAYERS='${NUM_LAYERS}'
export HIDDEN_SIZE='${HIDDEN_SIZE}'
export FFN_HIDDEN_SIZE='${FFN_HIDDEN_SIZE}'
export NUM_ATTENTION_HEADS='${NUM_ATTENTION_HEADS}'
export SEQ_LENGTH='${SEQ_LENGTH}'
export NUM_EXPERTS='${MODEL_NUM_EXPERTS}'
export TENSOR_MODEL_PARALLEL_SIZE='${TENSOR_MODEL_PARALLEL_SIZE}'
export EXPERT_MODEL_PARALLEL_SIZE='${FULL_EP_SIZE}'
export EXPERT_TENSOR_PARALLEL_SIZE='${EXPERT_TENSOR_PARALLEL_SIZE}'
export NONUNIFORM_MODE=ep
export NONUNIFORM_EP_DDP_APPROACH=nccl
export NONUNIFORM_EP_TOPOLOGY='${FULL_TOPOLOGY_UNIT} ${REDUCED_TOPOLOGY_UNIT}'
export MOE_TOKEN_DISPATCHER_TYPE='${TOKEN_DISPATCHER}'
export ENABLE_PYTORCH_PROFILER=1
export PROFILE_STEP_START=1
export PROFILE_STEP_END=3
export PROFILE_RANKS='${PROFILE_RANKS}'
export EXTRA_MEGATRON_ARGS='${extra_args}'
bash '${RUNNER}'
"
echo "[nep-distopt-smoke] $(date --iso-8601=seconds) training completed"

python3 - "${DRIVER_LOG}" "${CHECKSUM_DIR}" "${RUN_DIR}" "${OVERLAP_PARAM_GATHER}" \
    "${FULL_EP_SIZE}" "${REDUCED_EP_SIZE}" "${MODEL_NUM_EXPERTS}" "${TOKEN_DISPATCHER}" <<'PY'
import gzip
import json
import math
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
checksum_dir = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
expect_overlap_param_gather = bool(int(sys.argv[4]))
full_ep_size = int(sys.argv[5])
reduced_ep_size = int(sys.argv[6])
num_experts = int(sys.argv[7])
token_dispatcher = sys.argv[8]
world_size = full_ep_size + reduced_ep_size
text = log_path.read_text()

if not re.search(r"use_distributed_optimizer\s+\.+\s+True", text):
    raise RuntimeError("Run did not report use_distributed_optimizer=True")
reported_overlap = bool(re.search(r"overlap_param_gather\s+\.+\s+True", text))
if reported_overlap != expect_overlap_param_gather:
    raise RuntimeError(
        "Unexpected overlap_param_gather setting: "
        f"expected={expect_overlap_param_gather}, reported={reported_overlap}"
    )
if not re.search(rf"moe_token_dispatcher_type\s+\.+\s+{token_dispatcher}", text):
    raise RuntimeError(f"Run did not report token dispatcher {token_dispatcher}")
if token_dispatcher == 'flex' and not re.search(
    r"moe_flex_dispatcher_backend\s+\.+\s+hybridep", text
):
    raise RuntimeError("Flex run did not report the HybridEP backend")

placement_pattern = re.compile(
    r"initialize_nonuniform_ep_process_groups exit rank=(\d+) "
    r"local_ep_size=(\d+) ep_rank=(\d+) local_expert_indices=\[([^]]*)\]"
)
placements = {}
for match in placement_pattern.finditer(text):
    rank, ep_size, ep_rank = map(int, match.group(1, 2, 3))
    experts = [int(value.strip()) for value in match.group(4).split(',') if value.strip()]
    placements[(ep_size, ep_rank)] = (rank, experts)
expected_keys = {
    (ep_size, rank)
    for ep_size in (full_ep_size, reduced_ep_size)
    for rank in range(ep_size)
}
if set(placements) != expected_keys:
    raise RuntimeError(f"Unexpected placement keys: {placements}")
for ep_size in (full_ep_size, reduced_ep_size):
    experts = [
        expert
        for ep_rank in range(ep_size)
        for expert in placements[(ep_size, ep_rank)][1]
    ]
    if sorted(experts) != list(range(num_experts)):
        raise RuntimeError(f"EP{ep_size} did not place each logical expert exactly once: {experts}")

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
            raise RuntimeError(f"Non-finite checksum at iteration={iteration} rank={rank}")
        records.setdefault(iteration, {})[rank] = values
if len(records) < 3:
    raise RuntimeError(f"Expected at least three checksum iterations, got {records.keys()}")
for iteration, rows in records.items():
    if len(rows) != world_size:
        raise RuntimeError(
            f"Iteration {iteration} has {len(rows)} ranks, expected {world_size}"
        )

rank_for = {key: value[0] for key, value in placements.items()}
full_reference_rank = rank_for[(full_ep_size, 0)]
reduced_reference_rank = rank_for[(reduced_ep_size, 0)]
exact_fields = {7, 8, 12, 13, 14}


def assert_close(label, lhs, rhs, exact=False):
    if exact:
        if lhs != rhs:
            raise RuntimeError(f"{label}: exact mismatch {lhs} != {rhs}")
    elif not math.isclose(lhs, rhs, rel_tol=2e-6, abs_tol=1e-7):
        raise RuntimeError(f"{label}: mismatch {lhs} != {rhs}")


ordered_iterations = sorted(records)
for reference_rank in (full_reference_rank, reduced_reference_rank):
    # Expert fields validate the NEP-owned state directly. With native parameter-gather
    # overlap, dense model buffers are deliberately reused for gradients after backward,
    # so this observation point is not a full-parameter checkpoint. Require every step to
    # change expert state so a stale previous-iteration completion flag cannot pass.
    for previous_iteration, iteration in zip(ordered_iterations, ordered_iterations[1:]):
        if all(
            math.isclose(
                records[previous_iteration][reference_rank][index],
                records[iteration][reference_rank][index],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for index in (9, 10, 11)
        ):
            raise RuntimeError(
                "Distributed optimizer did not update expert state: "
                f"rank={reference_rank} iterations={previous_iteration}->{iteration}"
            )

for iteration, rows in sorted(records.items()):
    full_reference = rows[full_reference_rank]
    reduced_reference = rows[reduced_reference_rank]
    for field in range(9, 14):
        assert_close(
            f"iteration={iteration} cross-replica {fields[field]}",
            full_reference[field],
            reduced_reference[field],
            exact=field in exact_fields,
        )
    # Native overlap-param-gather reuses dense parameter buffers for gradients after
    # backward.  At this post-step observation point only dense shape metadata is stable;
    # the expert checks below validate the state owned and redistributed by NEP.
    dense_fields = (7, 8) if expect_overlap_param_gather else range(4, 9)
    for rank, row in rows.items():
        for field in dense_fields:
            assert_close(
                f"iteration={iteration} rank={rank} {fields[field]}",
                full_reference[field],
                row[field],
                exact=field in exact_fields,
            )
    for ep_size, reference_rank in (
        (full_ep_size, full_reference_rank),
        (reduced_ep_size, reduced_reference_rank),
    ):
        for ep_rank in range(ep_size):
            rank = rank_for[(ep_size, ep_rank)]
            for field in range(9, 14):
                assert_close(
                    f"iteration={iteration} EP{ep_size} rank={rank} {fields[field]}",
                    rows[reference_rank][field],
                    rows[rank][field],
                    exact=field in exact_fields,
                )

        first_rank = rank_for[(ep_size, 0)]
        per_expert_numel = rows[first_rank][14] / len(placements[(ep_size, 0)][1])
        local_total = 0
        for ep_rank in range(ep_size):
            rank = rank_for[(ep_size, ep_rank)]
            expected_local_numel = per_expert_numel * len(placements[(ep_size, ep_rank)][1])
            assert_close(
                f"iteration={iteration} EP{ep_size} rank={rank} has no dummy expert params",
                rows[rank][14],
                expected_local_numel,
                exact=True,
            )
            local_total += rows[rank][14]
        assert_close(
            f"iteration={iteration} EP{ep_size} local/replica expert numel",
            local_total,
            rows[reference_rank][13],
            exact=True,
        )

trace_files = (
    list(run_dir.rglob("*.pt.trace.json"))
    + list(run_dir.rglob("*.pt.trace.json.gz"))
    + list(run_dir.rglob("torch_profile/rank-*.json"))
    + list(run_dir.rglob("torch_profile/rank-*.json.gz"))
)
if len(trace_files) != world_size:
    raise RuntimeError(
        f"Expected one profiler trace per rank, got {len(trace_files)} for world {world_size}"
    )

if expect_overlap_param_gather:
    min_ep_size = min(full_ep_size, reduced_ep_size)
    owner_ranks = {
        rank_for[(ep_size, ep_rank)]
        for ep_size in (full_ep_size, reduced_ep_size)
        for ep_rank in range(min_ep_size)
    }
    for trace_path in trace_files:
        rank_match = re.search(r"rank-(\d+)", trace_path.name)
        if rank_match is None:
            raise RuntimeError(f"Could not recover rank from trace path {trace_path}")
        rank = int(rank_match.group(1))
        opener = gzip.open if trace_path.suffix == '.gz' else open
        with opener(trace_path, 'rt') as stream:
            events = json.load(stream)['traceEvents']
        names = [
            event.get('name')
            for event in events
            if event.get('ph') == 'X' and event.get('cat') == 'user_annotation'
        ]
        starts = names.count('nep_distopt_native_param_all_gather')
        finishes = names.count('nep_distopt_finish_native_param_all_gather')
        redistributions = names.count('nep_distopt_param_scatter')
        is_owner = rank in owner_ranks
        if is_owner:
            if starts == 0 or starts != finishes or finishes != redistributions:
                raise RuntimeError(
                    f"Rank {rank} has an invalid owner param-sync lifecycle: "
                    f"starts={starts}, finishes={finishes}, redistributions={redistributions}"
                )
        elif starts != 0 or finishes != 0 or redistributions == 0:
            raise RuntimeError(
                f"Rank {rank} has an invalid follower param-sync lifecycle: "
                f"starts={starts}, finishes={finishes}, redistributions={redistributions}"
            )

print(
    "[nep-distopt-smoke] PASS: owner-layout reduce-scatter/parameter all-gather, "
    "cross-replica updates, and all-rank profiler traces"
)
PY
