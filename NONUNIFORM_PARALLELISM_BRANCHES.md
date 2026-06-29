# Nonuniform Parallelism Branch Guide

Status date: 2026-06-03

This document records the current purpose, implementation shape, usage, and
remaining work for the three nonuniform-parallelism branches we have been using:

- `ntp-implementation-dev-pr`
- `heterogeneous-ep-nvshmem`
- `nep-ntp-shared-implementation`

The branch heads below were checked against `origin` on 2026-06-03. Use the
remote refs as the source of truth, especially for
`nep-ntp-shared-implementation`, where the local checkout may be stale.

## Remote Locations

Primary remote:

```bash
origin  git@github.com:daiyaanarfeen/Megatron-LM.git
```

Upstream remote:

```bash
upstream  https://github.com/NVIDIA/Megatron-LM.git
```

| Branch | Remote ref | Head commit | Notes |
| --- | --- | --- | --- |
| `ntp-implementation-dev-pr` | `origin/ntp-implementation-dev-pr` | `2087dc15e2dd23ac4918893b9ae134dfb3e3f949` | Opt-in NTP PR branch. Local stale ref `upstream/pr-4585` also points at this commit in this checkout, but the upstream remote no longer advertises that branch. |
| `heterogeneous-ep-nvshmem` | `origin/heterogeneous-ep-nvshmem` | `af4e630a8692effc6eea6f6396ce12700beb6aef` | Standalone heterogeneous EP branch with the three A/B/C approaches. This branch is now published under the same name on `origin`. |
| `nep-ntp-shared-implementation` | `origin/nep-ntp-shared-implementation` | `8728c3dffb5fa75bb482f5ab60885a9e189b04c6` | Shared NTP/NEP branch. Local branch in this checkout was behind the remote; use `origin/nep-ntp-shared-implementation`. |

Helpful checkout commands:

```bash
git fetch origin
git checkout ntp-implementation-dev-pr
git checkout heterogeneous-ep-nvshmem
git checkout nep-ntp-shared-implementation
```

If a local branch does not already exist:

```bash
git fetch origin ntp-implementation-dev-pr:ntp-implementation-dev-pr
git fetch origin heterogeneous-ep-nvshmem:heterogeneous-ep-nvshmem
git fetch origin nep-ntp-shared-implementation:nep-ntp-shared-implementation
```

## Branch: `ntp-implementation-dev-pr`

### Motivation And Goals

`ntp-implementation-dev-pr` implements Nonuniform Tensor Parallelism, abbreviated
NTP. The motivation is to keep training running when one or more tensor-parallel
replicas cannot use the full healthy TP width. Typical examples are:

- a faulty or unavailable GPU inside one TP replica;
- nonuniform NVL domains, where one replica naturally has fewer GPUs than
  another;
- benchmarking a reduced-TP replica against healthy full-TP replicas without
  changing the default Megatron path.

The branch goal is a non-intrusive, opt-in NTP API. Standard Megatron DDP,
training loops, and model construction remain the default. A training script
must explicitly import NTP classes and wrap the model with the NTP DDP subclass.

The high-level design target is:

```python
from megatron.core.distributed.nonuniform_tp import (
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    initialize_nonuniform_tp_process_groups,
    ntp_init,
    ntp_map,
)
```

### Primary Files

The branch adds these core files:

- `megatron/core/distributed/nonuniform_tp.py`
- `megatron/core/distributed/README_NONUNIFORM_TP.md`
- `megatron/core/extensions/nonuniform_tp_transformer_engine.py`
- `tests/unit_tests/distributed/test_nonuniform_tp.py`
- `tests/unit_tests/extension/test_nonuniform_tp_transformer_engine.py`

### Implementation Summary

The implementation is centered on `NonuniformTPConfig` and
`NonuniformTPDistributedDataParallel`.

`NonuniformTPConfig` describes a nominal healthy TP size and the reduced TP
replicas:

- `tp_base`: TP width for healthy replicas.
- `tp_spares`: number of local TP slots removed from reduced replicas.
- `num_reduced_tp_dp_ranks`: number of DP replicas that use reduced TP when no
  explicit map is supplied.
- `non_active_ranks_per_dp`: optional map from `(dp_rank, cp_rank, pp_rank)` to
  local TP rank IDs that are not active.

The process-group helper `initialize_nonuniform_tp_process_groups()` is called
after normal Megatron model-parallel initialization. It rewrites the affected
TP, CP, and TP-CP groups for reduced replicas so only active local TP ranks are
used. Healthy replicas keep the full `tp_base` group.

The parameter mapping helpers `ntp_init()` and `ntp_map()` attach split metadata
to tensor-parallel parameters. Healthy full-TP ranks need this metadata so they
know how to gather gradients from extra TP ranks into core ranks and scatter the
synced gradients back.

The DDP implementation adds NTP-specific buffer and bucket-group behavior:

- `NonuniformTPParamAndGradBuffer` computes a DDP buffer layout that includes
  extra `side_grad` storage on healthy core ranks.
- `NonuniformTPParamAndGradBucketGroup` wraps normal DDP bucket groups and
  controls which ranks participate in DP gradient sync.
- Extra healthy TP ranks skip DP sync because they have no peer in reduced TP
  replicas.
- Healthy core ranks use `side_grad` to carry the extra-rank gradients that are
  folded into the core DP sync.
- Post-sync all-to-all resharding scatters synced gradients back to extra ranks
  before the normal optimizer step.

The optimized overlap behavior is in the DDP/bucket-group path:

- pending NTP reshard handles are waited before starting bucket DP sync;
- `finish_grad_sync()` launches async post-sync gradient resharding;
- post-sync reshard waits are deferred so earlier post-sync reshards can overlap
  with the final bucket reductions.

The branch also includes Transformer Engine extension support through
`nonuniform_tp_transformer_engine.py`, so TP-sharded TE layers can be mapped into
the same NTP split metadata.

### Usage

The README on the branch shows the expected opt-in usage:

```python
ntp_config = NonuniformTPConfig(
    tp_base=4,
    tp_spares=2,
    num_reduced_tp_dp_ranks=1,
    non_active_ranks_per_dp={(0, 0, 0): [2, 3]},
)

# Call after initialize_model_parallel(... tensor_model_parallel_size=4 ...).
initialize_nonuniform_tp_process_groups(ntp_config)
```

After model construction:

```python
for module in model.modules():
    if module.__class__.__name__ == "TransformerLayer":
        ntp_init(module, ntp_config)

ntp_map(model.embedding.word_embeddings, ntp_config, vocab_size)
ntp_map(model.output_layer, ntp_config, vocab_size)
```

Then wrap DDP with the opt-in class:

```python
ddp_model = NonuniformTPDistributedDataParallel(
    config=config,
    ddp_config=ddp_config,
    module=model,
    disable_bucketing=False,
    pg_collection=pg_collection,
    ntp_config=ntp_config,
)
```

Important usage constraints:

- the launcher still controls global-rank to physical-GPU placement;
- `non_active_ranks_per_dp` values are local TP slot IDs, not global ranks;
- use `overlap_grad_reduce=True` for the optimized path;
- keep bucket count small enough that post-sync all-to-all launch overhead does
  not dominate;
- validate rank placement before trusting performance numbers.

The branch README gives a specific mapping example for a TP2 plus TP4 layout:

```text
global ranks 0,1     -> reduced TP2 replica
global ranks 2,3,4,5 -> healthy TP4 replica
```

The helper path assumes contiguous nominal DP replicas. More topology-aware rank
generation was added later in `nep-ntp-shared-implementation`.

### Current Validation Status

The branch contains focused unit tests for:

- NTP config and rank behavior;
- buffer layout with `side_grad`;
- DDP bucket wrapping;
- Transformer Engine extension mapping;
- process-group creation ordering and userbuffer restoration fixes.

The branch was also used as the PR 4585 implementation base. In this checkout,
the local `upstream/pr-4585` tracking ref points at the same commit, but the
upstream remote no longer advertises a `pr-4585` branch.

### Remaining Work

Implementation work still worth doing:

- Add a first-class training entrypoint or documented launcher for NTP rather
  than requiring users to hand-write opt-in integration in every script.
- Bring the topology-aware rank generation from
  `nep-ntp-shared-implementation` back into this branch if this branch remains
  the standalone NTP source.
- Clarify and harden optimizer support. The path is safest when all ranks have
  synced local gradients before a non-distributed optimizer step. Distributed
  optimizer behavior should be explicitly validated before being advertised.
- Extend coverage to Mamba/SSM layers and other TP-sharded non-transformer
  modules.
- Add broader CP and PP coverage. The current branch documents CP-aware tuple
  keys, but most performance work focused on PP1.
- Add a standard HSG performance harness and profiler trace workflow on this
  branch, or rely on the shared branch for benchmarks.

Testing work still needed:

- Multi-node correctness runs with realistic model sizes.
- Training parity against uniform TP baselines over longer runs.
- Pytorch profiler traces showing pre-sync and post-sync all-to-all overlap.
- Bucket-count sensitivity tests.
- Failure-case tests for bad rank maps, inconsistent CP slices, missing
  `ntp_map()` calls, and incorrect physical rank placement.

## Branch: `heterogeneous-ep-nvshmem`

### Motivation And Goals

`heterogeneous-ep-nvshmem` is the standalone branch for heterogeneous expert
parallelism, now more consistently described as nonuniform EP or NEP. The goal
is to compare and optimize gradient synchronization when different MoE replicas
have different expert-parallel widths, such as:

```text
EP8 / EP6 with TP2 CP2
EP16 / EP12 with TP2 CP2
EP32 / EP28 with TP2 CP2
```

The explicit benchmark goal was to compare nonuniform EP against a uniform EP
baseline while preserving standard Megatron training semantics as much as
possible. For example, a reduced 12-GPU replica should use a proportionally
smaller local sample count than a 16-GPU replica.

This branch contains all three historical NEP gradient-sync approaches:

- Approach A: NCCL baseline.
- Approach B: NVSHMEM pipelined implementation.
- Approach C: phased NCCL `all_to_all_single` implementation.

Unlike `ntp-implementation-dev-pr`, this branch is not purely a shared API
cleanup branch. It is a focused NEP implementation and benchmarking branch.

### Primary Files

Core implementation files:

- `megatron/core/distributed/heterogeneous_ep.py`
- `megatron/core/distributed/pipelined_reshard_collective.py`
- `megatron/core/parallel_state.py`
- `megatron/core/distributed/distributed_data_parallel.py`
- `megatron/core/distributed/distributed_data_parallel_config.py`
- `megatron/core/distributed/param_and_grad_buffer.py`
- `megatron/core/transformer/moe/moe_layer.py`
- `megatron/core/transformer/moe/token_dispatcher.py`
- `pretrain_gpt_heterogeneous_ep.py`

Test and benchmark files:

- `tests/unit_tests/distributed/test_heterogeneous_ep_grad_sync.py`
- `tests/unit_tests/distributed/test_heterogeneous_ep_training.py`
- `tests/unit_tests/transformer/moe/test_heterogeneous_token_dispatcher.py`
- `tests/unit_tests/distributed/bench_heterogeneous_ep_grad_sync.py`
- `scripts/heterogeneous_ep/*.sh`
- `tools/heterogeneous_ep/*.py`

HSG and profiling helper files were also added under `tools/`.

### Implementation Summary

The user-facing API is:

```python
from megatron.core.distributed.heterogeneous_ep import (
    HeterogeneousEPConfig,
    HeterogeneousEPDistributedDataParallel,
)

model = HeterogeneousEPDistributedDataParallel(
    config=config,
    ddp_config=ddp_config,
    module=module,
    heterogeneous_ep_config=HeterogeneousEPConfig(approach="nvshmem"),
)
```

`HeterogeneousEPConfig` contains:

- `approach`: one of `"nccl"`, `"nvshmem"`, or `"phased"`;
- `num_pipeline_chunks`: optional chunk count for the NVSHMEM path.

The branch also retains compatibility with legacy DDP flags:

- `use_pipelined_ep_reshard`
- `use_phased_ep_reshard`
- `num_ep_reshard_pipeline_chunks`

`HeterogeneousEPDistributedDataParallel` calls regular Megatron DDP first, then
replaces only `expert_parallel_bucket_groups` with EP-aware bucket groups.
Dense parameters continue to use normal DDP behavior.

The heterogeneous EP process groups and expert placement are created through
`initialize_heterogeneous_model_parallel()` in `parallel_state.py`. The branch
adds heterogeneous rank generation for attention and expert dimensions, plus
placement metadata used by token routing and expert-gradient synchronization.

The opt-in GPT entrypoint is `pretrain_gpt_heterogeneous_ep.py`. It is based on
standard `pretrain_gpt.py` providers and installs entrypoint-local hooks:

- replaces `parallel_state.initialize_model_parallel` with heterogeneous model
  parallel initialization;
- replaces the training module's DDP class with a shim that constructs
  `HeterogeneousEPDistributedDataParallel`;
- adds CLI flags for heterogeneous EP topology and approach selection.

### Approach A: NCCL Baseline

Approach A is the simple NCCL baseline:

1. Gather expert gradients across the local EP group.
2. Run expert-data-parallel allreduce on eligible owner/leader ranks.
3. Scatter the synced gradient slices back across EP ranks.

This path is intentionally simple. It uses standard collectives and avoids the
NVSHMEM runtime. The latest branch state avoids persistent local-EP-sized gather
buffers for this path because those buffers can dominate memory on large-expert
benchmarks.

Approach A is useful as:

- a correctness reference;
- a low-dependency baseline;
- a memory comparison point against NVSHMEM.

### Approach B: NVSHMEM Pipelined Path

Approach B is the optimized NVSHMEM path. Its purpose is to turn the gather,
EDP allreduce, and scatter sequence into a more pipeline-friendly operation.

Key implementation details:

- Uses `PipelinedReshardCollective` and
  `_HeterogeneousEPPipelinedReshardCollective`.
- Uses NVSHMEM symmetric buffers, signals, acks, and quiet operations.
- Supports multi-leader ring allreduce and interleaved expert placement.
- Uses per-expert gradient slices for layer-aligned expert buckets.
- Uses a monotonic signal-base allocator to avoid reusing signal epochs across
  buckets.
- Records CUDA events from the NVSHMEM stream so Megatron bucket finishing can
  wait without forcing every operation to become synchronous.
- Uses layer-aligned expert bucket groups for the optimized path.

The branch includes several iterations of this approach:

- early NVSHMEM put and signal experiments;
- separate slot buffers and NVLS workarounds;
- concurrent puts and batched quiet;
- team barrier optimization;
- double-buffered staging;
- interleaved expert placement;
- tuned slot-size scripts and HSG stress benchmark scripts.

Known tradeoffs:

- NVSHMEM symmetric staging memory can be substantial.
- Current conservative slot allocation uses `2 * max_ep` style staging in the
  optimized path.
- Reducing staging slots is possible, but requires a credit/ack protocol or
  route-coloring so followers do not overwrite leader receive slots.

### Approach C: Phased NCCL `all_to_all_single`

Approach C is the phased NCCL implementation:

1. Use EP-group `all_to_all_single` for gather.
2. Use EDP allreduce on leaders/owners.
3. Use reverse EP-group `all_to_all_single` for scatter.

This path is behavior-preserving relative to the conceptual NCCL reshard
approach, but more structured than Approach A. In the latest pushed state, the
phased path also requires layer-aligned expert buckets, like the NVSHMEM path.

This branch did not turn Approach C into a fully overlapped async path. It is
primarily a correctness and implementation-comparison baseline for NCCL
all-to-all style resharding.

### Token Routing And Placement

The branch modifies MoE token routing to account for heterogeneous expert
placement. The rank generator computes local expert indices and placement maps.
The token dispatcher uses this metadata so routed tokens are sent to the rank
that physically owns the selected expert in a nonuniform EP group.

The placement strategy supports interleaved expert assignment for arbitrary
`ep / min_ep` ratios. This was needed for cases like EP16/EP12 where the wider
replica has follower ranks holding offloaded expert slices from owner ranges.

### Usage

The direct Python API:

```python
parallel_state.initialize_heterogeneous_model_parallel(
    tensor_model_parallel_size=2,
    context_parallel_size=2,
    num_tp_cp_per_replica=[4, 3],
    expert_tensor_parallel_size=1,
    num_moe_experts=96,
    heterogeneous_ep_approach="nvshmem",
)

ddp_model = HeterogeneousEPDistributedDataParallel(
    config=config,
    ddp_config=ddp_config,
    module=model,
    heterogeneous_ep_config=HeterogeneousEPConfig(approach="nvshmem"),
)
```

The preferred training entrypoint on this branch is:

```bash
torchrun --nproc_per_node=4 --nnodes=7 \
  pretrain_gpt_heterogeneous_ep.py \
  --tensor-model-parallel-size 2 \
  --context-parallel-size 2 \
  --expert-tensor-parallel-size 1 \
  --num-experts 96 \
  --heterogeneous-ep-ddp-approach nvshmem \
  --heterogeneous-ep-num-tp-cp-per-replica 4 3 \
  ...standard pretrain_gpt.py arguments...
```

The important CLI flags are:

- `--heterogeneous-ep-num-tp-cp-per-replica`
- `--heterogeneous-ep-ddp-approach {nccl,nvshmem,phased}`
- `--heterogeneous-ep-num-pipeline-chunks`

HSG examples:

```bash
sbatch scripts/heterogeneous_ep/prepare_hsg_cached_image.sh

IMAGE=/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-nvshmem-megatron-het-ep.sqsh \
INSTALL_NVSHMEM=0 \
sbatch scripts/heterogeneous_ep/run_standard_training_ep8_6_compare.sh
```

Stress benchmark:

```bash
IMAGE=/lustre/fsw/portfolios/coreai/users/darfeen/pyt25.10-nvshmem-megatron-het-ep.sqsh \
RUN_HETERO=1 RUN_UNIFORM=1 \
sbatch scripts/heterogeneous_ep/run_ep16_12_nvshmem_uniform_stress.sh
```

The branch also includes EP32 uniform baseline and NTP Mamba helper scripts under
`tools/` for HSG benchmarking.

### Current Validation Status

The branch includes unit and integration-style tests for:

- heterogeneous rank generation;
- expert placement;
- token dispatcher behavior;
- heterogeneous EP gradient sync;
- heterogeneous EP training parity;
- NVSHMEM put/signal and ring checks;
- HSG benchmark scripts for 8, 12, 24, 28, and 32 GPU style experiments.

Observed benchmark history from this branch and related HSG runs:

- Small EP8/EP6 examples were used for correctness and approach comparison.
- EP16/EP12 NVSHMEM stress workloads were used to expose slot-size and overhead
  issues.
- Larger EP32 uniform baseline work moved toward DeepEP/HybridEP for optimized
  production-style baseline measurement.

### Remaining Work

Implementation work still needed:

- Decide whether this branch remains standalone or is ported into the shared
  `nonuniform_ep.py` API from `nep-ntp-shared-implementation`.
- If porting, add an approach selector to `NonuniformEPConfig` and move the
  A/B/C logic behind the shared DDP/bucket wrapper.
- Reduce NVSHMEM staging memory. The most promising options are static
  route-coloring, a smaller per-leader receive pool, or a credit-based fixed
  slot protocol.
- Make Approach C asynchronous if it remains relevant. The current phased path
  is structured but not a fully optimized overlapped implementation.
- Confirm whether the latest branch state exits cleanly with the current HSG
  container and NVSHMEM runtime.
- Revisit distributed optimizer support. The safest design is still local
  optimizer semantics after scatter-back, but that should be stated and tested.
- Minimize or isolate generic Megatron file changes if this branch is prepared
  for upstreaming. This branch changes `parallel_state.py`, MoE token dispatch,
  DDP config, and DDP/buffer internals.

Testing work still needed:

- Re-run unit tests after the latest pushed commit.
- Re-run A/B/C correctness on supported topologies:
  - 12 GPU, `k=[2,4]`
  - 28 GPU, `k=[4,4,6]`
  - 32 GPU, `k=[4,4,8]`
- Re-run training parity for A/B/C against uniform EP baselines.
- Collect profiler traces for each approach on the same workload.
- Compare NVSHMEM against NCCL and phased on a compute-heavy workload where
  NVSHMEM should show a scheduling advantage.
- Add teardown tests for NVSHMEM resources and container/runtime compatibility.

## Branch: `nep-ntp-shared-implementation`

### Motivation And Goals

`nep-ntp-shared-implementation` is the shared implementation branch. Its goal is
to make NTP and NEP use the same opt-in style:

- no generic training-loop modifications by default;
- script-local DDP wrapper selection;
- shared rank-generation, bucket scheduling, and helper machinery;
- consistent `nonuniform_*` naming;
- standard GPT entrypoint derived from `pretrain_gpt.py`.

This branch is the closest match to the intended API style of
`ntp-implementation-dev-pr`, but extended to support NEP.

Important distinction:

`nep-ntp-shared-implementation` does not contain the old NEP Approach A/B/C
selector. It contains a single NEP implementation: P2P gradient ownership
transfer to owner ranks, owner-side DP sync, and scatter-back for local optimizer
semantics.

### Primary Files

Core implementation files:

- `examples/nonuniform/pretrain_gpt_nonuniform.py`
- `examples/nonuniform/README.md`
- `megatron/core/distributed/nonuniform_common.py`
- `megatron/core/distributed/nonuniform_tp.py`
- `megatron/core/distributed/nonuniform_ep.py`
- `megatron/core/extensions/nonuniform_tp_transformer_engine.py`

Tests and HSG scripts:

- `tests/unit_tests/distributed/test_nonuniform_tp.py`
- `tests/unit_tests/distributed/test_nonuniform_ep.py`
- `tests/unit_tests/distributed/test_nonuniform_topology.py`
- `tests/unit_tests/extension/test_nonuniform_tp_transformer_engine.py`
- `scripts/nonuniform/run_hsg_ep32_28_tp2cp2_compare.sh`

### Shared Machinery

`nonuniform_common.py` provides common helpers for both NTP and NEP:

- runtime config registration for NEP token routing;
- expert placement helpers;
- expert-to-EP-rank map construction;
- `NonuniformEPRankGenerator`;
- topology-aware group generation helpers;
- ordered bucket-group scheduling;
- utilities for filtering kwargs and handling global ranks;
- common padding/layout helpers;
- `all_to_all_with_output_views()` for non-contiguous all-to-all outputs.

The shared scheduling helpers are important because both NTP and NEP need to
preserve deterministic bucket ordering while still launching communication as
soon as a bucket is ready.

### NTP Implementation In The Shared Branch

The NTP implementation in this branch is based on the PR 4585 branch but adds
topology-aware rank generation.

`NonuniformTPConfig` includes the PR fields plus:

- `tp_domain_sizes`: active TP size per contiguous replica/domain.
- `topology_rank_metadata`: runtime mapping from global rank to topology
  coordinates.

Topology-aware NTP lets the entrypoint create TP groups inside contiguous
rank blocks, such as:

```bash
torchrun --nproc-per-node 4 --nnodes 3 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode tp \
  --tensor-model-parallel-size 4 \
  --context-parallel-size 1 \
  --nonuniform-tp-base 4 \
  --nonuniform-tp-spares 2 \
  --nonuniform-tp-domain-sizes 2 4 4 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

Here the first TP domain is reduced TP2 and the next two domains are healthy
TP4. Values must currently be either `tp_base` or `tp_base - tp_spares`, which
keeps the existing reduced/full resharding semantics.

The DDP and buffer mechanics remain the NTP mechanics from the PR branch:

- NTP split metadata on TP parameters;
- side-grad storage on healthy core ranks;
- extra-rank DP sync skipping;
- async post-sync all-to-all reshard;
- delayed waits for overlap.

### NEP Implementation In The Shared Branch

NEP is implemented in `nonuniform_ep.py` as opt-in gradient ownership transfer.

`NonuniformEPConfig` contains:

- `runtime_config`: runtime process-group and placement metadata;
- `expert_owner`: optional explicit expert to owner EP-rank map;
- `expert_name_pattern`: pattern for finding expert IDs in parameter names;
- `require_owner_local_expert`: safety check for local optimizer semantics;
- P2P tag bases for gather and scatter.

The key class is `NonuniformEPDistributedDataParallel`. It calls regular DDP,
then replaces expert bucket groups with `NonuniformEPParamAndGradBucketGroup`
instances.

The runtime behavior is:

1. Each expert bucket group computes one or more transfer plans.
2. Non-owner ranks pack their expert gradient slices into persistent staging
   buffers.
3. Non-owner ranks send those packed gradients to the owner EP rank with
   `dist.isend`.
4. Owner ranks receive with `dist.irecv` and accumulate into normal contiguous
   DDP gradient storage.
5. Owner ranks start ordinary expert-data-parallel DDP sync after required
   gathers complete.
6. Owner ranks scatter synced gradients back to source ranks.
7. Every rank has local synced gradients before the normal non-distributed
   optimizer step.

This design intentionally targets local optimizer semantics. The benchmark
entrypoint rejects `--use-distributed-optimizer`.

Important NEP details:

- P2P is scoped to the EP group, not global ranks.
- A dedicated NEP transfer communicator is used.
- Pre-sync gathers are ordered so owner allreduces start deterministically.
- Post-sync transfers are overlapped and their waits are deferred.
- Persistent staging buffers avoid repeated dynamic allocation.
- Transfers can be grouped by logical slot and bucket.
- Grouped-GEMM slot layouts are supported.
- Synthetic NEP buckets are used where the bucket plan needs an owner-side
  transfer/scheduling object that does not correspond one-to-one to a normal
  physical expert bucket. These are implementation artifacts for consistent
  ordering and ownership transfer, not separate model parameters.

### Token Routing In The Shared Branch

NEP token routing is handled through runtime expert placement metadata in
`nonuniform_common.py`.

The placement helper builds an `expert_id -> ep_rank` map. Token dispatch uses
that map so a token routed to a global expert is sent to the EP rank that
physically owns that expert.

For topology mode:

```bash
--nonuniform-ep-num-tp-cp-per-replica 8 7
```

with `TP2 CP2 ETP1`, the branch derives EP32/EP28 replicas:

- `TP * CP = 4`
- first replica: `8 * 4 / 1 = EP32`
- second replica: `7 * 4 / 1 = EP28`

The initializer:

- creates nonuniform EP process groups;
- computes interleaved expert placement;
- registers runtime placement before model construction;
- uses the same runtime config for token routing and DDP gradient ownership
  transfer.

Manual placement is also supported:

```bash
--nonuniform-ep-placement-path /path/to/ep_placement.json
--nonuniform-ep-expert-owner-path /path/to/expert_owner.json
```

The placement file is a JSON list with one entry per EP rank. Each entry lists
the global expert IDs physically present on that EP rank.

### Usage

The shared entrypoint is:

```bash
examples/nonuniform/pretrain_gpt_nonuniform.py
```

It supports:

```bash
--nonuniform-mode none
--nonuniform-mode tp
--nonuniform-mode ep
```

NTP example:

```bash
torchrun --nproc-per-node 8 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode tp \
  --nonuniform-tp-base 8 \
  --nonuniform-tp-spares 2 \
  --nonuniform-tp-num-reduced-dp-ranks 1 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

NEP topology example:

```bash
torchrun --nproc-per-node 4 --nnodes 15 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode ep \
  --tensor-model-parallel-size 2 \
  --context-parallel-size 2 \
  --expert-tensor-parallel-size 1 \
  --nonuniform-ep-num-tp-cp-per-replica 8 7 \
  --num-experts 224 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

HSG benchmark:

```bash
ssh hsg-1
cd /lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port
sbatch scripts/nonuniform/run_hsg_ep32_28_tp2cp2_compare.sh
```

The HSG launcher requests `--segment=16` and compares:

- uniform EP32, TP2, CP2, ETP1 on 64 ranks;
- nonuniform EP32/EP28, TP2, CP2, ETP1 on 60 ranks.

Default batch sizes are proportional:

- `UNIFORM_GBS=32`
- `NONUNIFORM_GBS=30`

### Current Validation Status

The branch includes unit tests for:

- nonuniform TP;
- nonuniform EP;
- nonuniform topology/rank generation;
- Transformer Engine NTP extension behavior.

The HSG benchmark launcher was added for EP32/EP28 TP2 CP2 comparison, including
optional Pytorch profiler traces. The observed performance work showed that the
initial P2P NEP path still had meaningful overhead versus a uniform baseline,
especially when uniform EP communication itself was not well hidden. Subsequent
work focused on transfer ordering, grouping, and persistent buffers.

### Remaining Work

Implementation work still needed:

- Port the three old NEP approaches from `heterogeneous-ep-nvshmem` if this
  branch is intended to become the unified NEP branch:
  - Approach A: NCCL gather/allreduce/scatter baseline.
  - Approach B: NVSHMEM pipelined implementation.
  - Approach C: phased all-to-all implementation.
- Add an explicit NEP approach enum/config, analogous to
  `HeterogeneousEPConfig(approach=...)`, if multiple NEP paths are ported.
- Decide whether the current P2P owner-transfer path is a fourth approach,
  the default local-optimizer path, or a temporary stepping stone.
- Improve NEP communication coalescing. The current branch has grouping by
  logical slot and bucket, but there is still room to reduce kernel launches
  with layer-level or peer/layer-slot packing.
- Investigate all-to-all based transfer for grouped NEP packets, while keeping
  enough granularity for overlap with owner allreduce.
- Keep improving overlap:
  - pre-sync gathers should be nonblocking where safe;
  - owner DP sync should start as soon as required gathers complete;
  - post-sync scatter waits should be deferred until the last useful point.
- Revisit distributed optimizer support. The current entrypoint intentionally
  rejects `--use-distributed-optimizer`; supporting it would require explicit
  optimizer-state placement and ownership semantics.
- Reduce or remove synthetic bucket complexity if layer-level packing provides a
  cleaner bucket model.
- Decide whether DeepEP/HybridEP baseline support belongs in this branch or only
  in benchmark scripts.

Testing work still needed:

- Full multi-node correctness and training parity for NTP and NEP on HSG.
- Standard profiler-output capture for both uniform and nonuniform runs.
- Performance comparisons against compute-heavy uniform baselines where EP, TP,
  CP, and DP communication are mostly hidden.
- EP32/EP28 benchmark reruns after communication coalescing changes.
- NTP Mamba-layer benchmarks using the merged PR 4585 NTP implementation.
- Tests for manual NEP placement files and explicit expert-owner overrides.
- Tests for grouped-GEMM expert-name matching and synthetic bucket ordering.
- Tests for invalid topology:
  - missing experts;
  - duplicated expert placement;
  - non-divisible `num_experts`;
  - mismatched `--nonuniform-ep-min-size`;
  - rank groups that cross physical NVL domains unexpectedly.

## Cross-Branch Relationship

The branches are complementary but not equivalent.

`ntp-implementation-dev-pr` is the clean standalone NTP branch. It has the
opt-in API style that NEP should follow.

`heterogeneous-ep-nvshmem` is the branch with the most complete historical NEP
approach coverage. It has A/B/C and the optimized NVSHMEM work, but it is less
aligned with the shared NTP/NEP API structure.

`nep-ntp-shared-implementation` is the branch with the best shared architecture.
It has the combined opt-in GPT entrypoint, common rank-generation helpers, NTP
topology improvements, NEP token routing, and P2P owner-transfer NEP. It does
not yet have A/B/C ported from `heterogeneous-ep-nvshmem`.

The most direct path to a unified branch is:

1. Start from `origin/nep-ntp-shared-implementation`.
2. Keep its `examples/nonuniform/pretrain_gpt_nonuniform.py` entrypoint and
   `nonuniform_common.py` shared helpers.
3. Port the useful A/B/C NEP implementations from
   `origin/heterogeneous-ep-nvshmem` into `nonuniform_ep.py`.
4. Preserve `ntp-implementation-dev-pr` NTP semantics and tests.
5. Add an explicit NEP approach selector.
6. Re-run correctness and performance on the same HSG workloads.

## Open Decisions

- Should the current P2P owner-transfer NEP path remain as the default NEP path,
  or should NVSHMEM become the default for performance?
- Should the unified API expose both `NonuniformEPConfig(approach=...)` and a
  lower-level transfer policy, or only an approach enum?
- Should distributed optimizer support be deferred explicitly, or implemented
  with owner-only optimizer semantics?
- Should all HSG benchmark scripts live in branch-specific `scripts/` folders,
  or should they be centralized under `tools/`?
- Should generated profiler output stay only in local `hsg_profile_traces/`, or
  should selected traces be uploaded to a separate artifact store?

