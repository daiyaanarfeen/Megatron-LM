# NEP: Algorithm, Systems, and Implementation

This document describes the expert-specific contribution, not a new collective
or optimizer algorithm. The common-owner reshard/synchronize/redistribute
principle is inherited from [NTP](https://arxiv.org/html/2504.06095v1).
NEP extends it to whole-expert identity, nondivisible expert placement, fused
dispatch layouts, and native Megatron distributed-optimizer integration.

## Algorithmic Contribution

Notation: `E` is the true expert count; `n[r]` is replica `r`'s EP degree;
`k = min(n)` is the number of logical owners per replica. A parameter is keyed
by `(MoE module, parameter slot, global expert ID)`, not its physical bucket
offset. Other parallel coordinates are held fixed in this pseudocode.
The current placement requires `E >= max(n)`.

```text
BUILD_LAYOUTS(E, n):
    k = min(n)
    O = balanced_contiguous_partition(range(E), k)
    # Owner i denotes the SAME real experts on every replica.
    owner_slots = pad_each_row(O, ceil(E / k), EMPTY)

    for replica r:
        target[i] = floor(E / n[r]) + (i < E mod n[r])
        for owner i < k:
            physical[r, i] = first target[i] experts of O[i]
        surplus = real experts not retained by these owners
        assign surplus to follower ranks k .. n[r]-1 to meet target
        # Nondivisible case: sequential balanced fill.
        # Divisible case: existing striped/round-robin follower policy.

        assert each real expert occurs exactly once in physical[r]
        dispatch_slots[r] = pad_each_row(physical[r], ceil(E / n[r]), EMPTY)
        apply backend-required alignment using EMPTY slots only
        build global-expert -> (physical rank, local slot) lookup

    for owner i:
        EDP[i] = corresponding owner i from every replica
    build matching owner bucket layouts using canonical parameter keys
    allocate reusable physical-to-owner transfer buffers

DISPATCH(tokens, replica r):
    routes = router(tokens, expert_ids=range(E))
    backend_routes = reindex routes into dispatch_slots[r]
    mask every EMPTY slot; invoke native Flex dispatch
    execute only real local expert parameters

ON_NATIVE_GRAD_READY(parameter p):
    accumulate p's gradient using native Megatron semantics
    if not final gradient-accumulation microbatch: return
    mark p ready in its canonical expert bucket
    for newly ready bucket b in legal collective order:
        asynchronously GATHER physical gradients into logical-owner layout
        make owner EDP consume the completed Gather layout
        launch native EDP synchronization in canonical bucket order
        # DistOpt: native reduce-scatter. Regular optimizer: all-reduce.
        # Preserve configured loss/gradient scaling for unequal local batches.

FINISH_BACKWARD_AND_UPDATE():
    complete pending gradient synchronization before optimizer consumption
    if distributed_optimizer:
        native optimizer updates sharded logical-owner parameters/state
        native parameter all-gather materializes updated owner parameters
        through the native forward pre-hook, redistribute owner parameters
            to physical expert holders before the corresponding layer uses them
    else:
        redistribute synchronized owner gradients to physical holders
        native regular optimizer updates physical expert parameters
```

This is dependency-level pseudocode: it does not prescribe a host wait after
every Gather, promise all communication is hidden, or replace the implementation's
NCCL ordering and buffer-lifetime safeguards. Native dense-parameter synchronization
and optimization remain on their existing paths. A follower has no authoritative
expert optimizer state in the DistOpt path, but still participates in dense
optimization and physical expert computation.

### Correctness Invariants

- The router and model retain exactly `E` real experts on every replica.
- Each real expert has exactly one physical holder per replica.
- Corresponding owners synchronize the same parameter keys and compatible layouts.
- Empty dispatch/communication slots create no trainable dummy experts.
- Gather precedes its EDP; optimizer consumption follows gradient completion;
  parameter redistribution completes before the physical expert is used.
- Unequal batch sizes use existing normalization, not an unweighted average of
  independently normalized replica gradients.

## Figures

The dedicated `nep_expert_contributions.png` figure makes the first two
contributions explicit: E3 moves between physical holders but synchronizes
under the same canonical key; the EP3 dispatch adapter presents 12 virtual
slots for 10 real experts without extending the router or trainable model.
The original systems figure provides the complementary DistOpt lifecycle.

1. **Systems:** physical expert placement differs across replicas, while owner
   layouts match across replicas. The example uses 10 real experts, EP4/EP3,
   and expert TP1; it is illustrative, not a benchmark. Owner rows are gradient
   and parameter communication layouts, not replicated full optimizer state.
2. **Implementation:** NEP replaces the expert synchronization adapter while
   retaining native DDP hooks, buffers, collective machinery, and optimizer.
   Arrows show dependencies and integration points, not measured overlap.

Rendered PNG/PDF files:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_contributions_20260911/`.
Reproduce with `python3 scripts/nonuniform/plot_nep_contributions.py`.

## Source Basis

Verified against the benchmark worktree
`/home/darfeen/Megatron-LM-nep-main-20260817` (HEAD `99ae92360`, including its
existing working-tree implementation), not a claim about every historical branch:

- `megatron/core/distributed/nonuniform_common.py`:
  `compute_nonuniform_ep_owner_expert_slots`,
  `compute_nonuniform_ep_expert_placement`, `compute_nonuniform_ep_dispatch_slots`.
- `megatron/core/distributed/nonuniform_ep.py`:
  `NonuniformEPDistributedDataParallel`,
  `NonuniformEPNCCLParamAndGradBucketGroup`,
  `_configure_nep_distributed_optimizer_buffers`, `register_grad_ready`,
  `_start_nep_nccl_owner_edp_reduce_contexts`, `finish_param_sync`.
- `megatron/core/transformer/moe/token_dispatcher.py`:
  `_initialize_nonuniform_flex_dispatcher`, `_initialize_nonuniform_metadata`.
- `megatron/core/optimizer/__init__.py`: native optimizer construction with
  logical-owner parameters and buffers.

Scope: this does not claim FSDP/checkpoint compatibility or all Flex backends;
the current nonuniform Flex adapter explicitly rejects NCCL-EP. The validated
feature combination discussed here is HybridEP, nondivisible experts, and DistOpt.
