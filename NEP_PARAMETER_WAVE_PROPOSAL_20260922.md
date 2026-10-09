# Proposal: reorder parameter broadcasts into disjoint-participant waves

## Finding and scope

There is a plausible small fix for the remaining cross-follower chain. No production
change or GPU job was made during this investigation. This is a proposal and a
trace-conditioned estimate, not a demonstrated speedup or a guaranteed bound.

Current parameter delivery iterates owner IDs in ascending order inside
`NonuniformEPNCCLParamAndGradBucketGroup._get_nep_distopt_param_transfer_plan`.
Shared followers connect that order into a long path across the full replica.
Use a deterministic first-fit partition of the existing transfer membership sets,
then traverse those waves in the cached plan. In the measured EP16/12 placement:

```
wave 0: owners 0, 3, 6, 9
wave 1: owners 1, 4, 7, 10
wave 2: owners 2, 8
wave 3: owners 5, 11
```

Members of each wave are disjoint. These are ordering classes, not new global
barriers: each rank simply traverses its participating transfers in this order.
Shared-rank dependencies always advance wave index, removing the 12-owner chain
without introducing a cycle among redistribution operations.

The gradient path already contains the same disjoint-membership scheduling algorithm
in `_start_nep_nccl_process_group_dispatch_batch` (around lines1914-1940). The parameter
path can use that scheduling pattern once while constructing its cached plan.
Leave gradient scheduling unchanged. Expected implementation scope is roughly20-30
added/changed production lines in `distributed/nonuniform_ep.py`, plus focused tests.
Exact LOC is not established until implementation.

## Why this differs from rejected parallel-owner streams

Keep the single parameter transfer stream per GPU, current staging buffers, process
groups, payloads, local copies, own-AG dependency, and parameter-ready waits. Do not
introduce multiple simultaneous NEP transfer kernels on a follower. Distinct GPUs
can communicate concurrently when their groups are disjoint. Steady-state Python
iteration and collective counts remain unchanged; schedule construction is cached.

The rejected multi-stream variant increased follower scan residency from2.099 to
34.876ms and regressed first forward13.154ms. This proposal avoids its additional
per-GPU concurrent NCCL work, but cannot guarantee freedom from aggregate-link,
memory, or native-kernel interference. Those require an actual profiled A/B.

Membership comes from `_nep_nccl_owner_transfer_ranks(owner)`, before local rank
filtering, so all participants derive the same order. The algorithm does not assume
EP16/12, divisibility, or a fixed follower count. Reduced replicas with singleton
transfers retain local copies; ownership, dummy slots, Flex dispatch, native DistOpt,
and physical/owner parameter layouts are not changed. GPU correctness across these
features is still a required validation step, not established by this analysis.

## Trace-conditioned estimate

Source: job3131835, candidate `ag_overlap_margin`, all28 rank CUDA-event profiles,
matched iterations20-27. Estimate script:
`scripts/nonuniform/estimate_nep_parameter_waves.py`.
Output: Lustre job3131835 `ag_redistribution_analysis/parameter_wave_estimate.json`.

The replay accounts for group participant readiness, serialized per-rank transfers,
measured post-all-input-ready completion intervals, intervening copy/submission gaps,
bucket consumption, native AG order, and critical-lane graph execution. It does not
sum rank-local waits. Replaying the recorded order reproduces first-forward endpoints
for all eight measured iterations; this is a model reconstruction check, not independent
validation of the counterfactual. All readiness/preparation fits have zero negative error.

| Scenario | First-F saving | First F | First-F parity | Iteration | Iteration parity |
|---|---:|---:|---:|---:|---:|
| Measured current |0|131.96ms|71.84%|560.52ms|86.86%|
| Reorder, freeze subsequent input/AG readiness |4.04ms|127.92ms|74.10%|556.48ms|87.49%|
| Reorder, propagate earlier full-replica compute into AG |12.63ms|119.33ms|79.44%|547.89ms|88.86%|
| Same propagation, post-ready broadcast intervals50% longer |11.41ms|120.55ms|78.63%|549.11ms|88.66%|

Healthy reference is94.792ms first F and486.866ms iteration. Per-step savings are
3.48-4.48ms for fixed readiness and11.78-13.02ms with full-replica propagation.
These are scenario ranges, NOT confidence intervals or guaranteed positive gains.

The coupled replay preserves the one-bucket-lookahead prefetch boundary: next AG
is submitted before waiting for the current bucket's delivery, and remains ordered
behind the preceding AG. Reduced-replica AG input readiness remains fixed to its
measured value; the model does not assume those peers advance when full ranks do.
It preserves measured critical-lane graph service and the first-forward suffix.
Subsequent phases are held unchanged when translating first-F savings to iteration
parity. It does not predict new resource contention, a different critical-rank compute
path, changed host launch starvation, or nonlinear link throughput. GPU validation
could therefore show less improvement, no improvement, or a regression.

Even erasing the entire37.165ms first-forward excess, with all later phases unchanged,
would yield only93.03% iteration parity. This proposal does not support a95% overall
parity promise; remaining final-backward/outside-phase costs would still matter.

## Validation before acceptance

1. Isolated branch from `ccb4b9726`; preserve the accepted branches.
2. Unit tests: all owners included exactly once, disjoint waves, identical global
   order before rank filtering, nondivisible/singleton memberships, cached schedule,
   and unchanged skip-next/force-sync lifecycle. Offline grouping already passed640
   arbitrary membership cases, which tests the proposed algorithm, not production code.
3. Four-GPU shared-follower fresh-value/reuse test, then short EP8/6 Flex/DistOpt/
   nondivisible training smoke with native model-EP and AG/RS overlap enabled.
4. EP16/12 healthy/current/candidate A/B, unchanged workload/placement/profiling.
   Compare all owner phase and iteration means, memory, and all-rank collective
   metadata. Require unchanged payloads/counts and at most one NEP redistribution
   collective active per GPU. Inspect native model-EP/scan kernels for the prior
   parallel-stream regression. Stop/revert if correctness or performance regresses.

No distributed-optimizer, dispatcher, gradient, or expert-placement refactor is
required for this first experiment. A packed full-replica all-to-all would be a
separate, more invasive proposal and is not included in the estimate above.
