# Expert-Width Parameter Delivery Audit

## Scope

Existing job3063815, all four cases, all120 rank profiles, iterations40-49.
Full-owner results average ranks0-5 inside each iteration, then ten iterations.
Reduced-owner ranks8-13 and full followers6-7 were checked separately. Healthy
also has ranks14-15 audited but not included in these cohorts. No new jobs or
runtime changes. Accepted runtime93d313df4, nonuniform_ep.py SHA256
653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7 matches
each case's launch manifest and the inspected worktree.

Comparison: F768 versus F3072 at H1280,E128,top-k6,sequence8192,MBS2,GA3,
EP8/8 healthy versus EP8/6 NEP, DistOpt+Flex/HybridEP. Model pattern has
15 MoE layers, 15 Mamba layers, 3 attention layers. Expert parameter bytes and
mathematical expert FLOPs both scale4x. Native physical expert buckets change
4->6, dense3->2; NEP logical parameter/gradient groups remain8. This is not an
iso-native-bucket experiment. No result is interpreted as a pure width-only
bandwidth experiment.

## 1. Measured Duration Scaling Is Not Proportional

NEP full owners; mean +/- sample SD across measured iterations, milliseconds.
Ordinary forward averages MB1/MB2 (zero-based); ordinary backward MB0/MB1.

| Quantity | F768 | F3072 | Ratio |
| --- | ---: | ---: | ---: |
| Grouped-forward GPU range sum / ordinary MB |14.005 +/-0.200|18.424 +/-0.198|1.316x|
| Grouped-backward GPU range sum / ordinary MB |12.163 +/-0.011|29.589 +/-0.013|2.433x|
| Parameter-redistribution GPU range sum / iteration |9.832 +/-0.122|33.951 +/-0.044|3.453x|

Healthy grouped-forward ranges also increase much less than4x:12.332->16.063ms.
NEP reduced-owner ordinary forward16.174->19.250ms, redistribution0.802->3.000ms.
The much smaller reduced redistribution range is expected to contain mostly
local owner-to-physical copies; it is not a cross-replica all-gather measurement.

These are operation elapsed ranges, NOT pure GEMM kernel busy time or
transfer-only service time. The grouped range includes internal launch spacing;
redistribution includes packing, broadcasts, unpacking and dependencies. Native
owner parameter all-gather is outside the redistribution wrapper. Therefore
the data establish unequal measured operation-duration scaling, not exact
compute/communication kernel scaling or the reason for that scaling.

## 2. NEP Redistribution Is Not Prefetched Behind Previous Expert Compute

In the executed `nonuniform_ep.py`:

- `start_param_sync`, line1749: starts native owner parameter all-gather and
  returns when overlap is enabled. It does not launch NEP redistribution.
- `finish_param_sync`, line1783: finishes native owner all-gather, calls NEP
  redistribution, then starts the next bucket's native parameter all-gather.
- `_scatter_nep_distopt_owner_params_to_physical_holders`, line1699: packing
  and unpacking use current-stream foreach copies; broadcasts have no
  `async_op=True` and no separate-stream prefetch scheduling.
- Native DDP `_finish_param_sync_for_bucket_group`, line542 in
  distributed_data_parallel.py, invokes finish from parameter-ready/pre-hook
  processing. This is demand-time publication, not eager redistribution.

Trace checks, for F3072/full owners:

- All480 redistribution ranges use the same CUDA stream as expert execution.
- Intersection with recorded expert-forward ranges is exactly0ms.
- For all420 noninitial calls, CPU submission occurs AFTER the preceding
  expert's GPU range completes, beyond3x recorded clock-error bound.
  Mean delay by bucket:0.732-0.939ms; minimum individual sample0.609ms.
- Noninitial redistribution GPU ranges average4.503-4.528ms each, followed
  by3.747-3.799ms before the next expert range starts. The latter is NOT
  automatically idle time; non-expert layers/model-EP work are uninstrumented.
- The same stream/nonoverlap/late-after-prior-expert checks hold for all four
  cases, reduced owners and full followers.

The native all-gather has a prefetch path in code, but its actual launch and
completion timestamps are not captured by this targeted event protocol. We
cannot certify its per-bucket lead time from these profiles. NEP redistribution,
however, definitively does not overlap the preceding expert computation here.

## 3. Overhead Repeats At Parameter-Bucket Boundaries, Not Layer Zero

Forward events have two grouped-linear ranges per MoE layer. Thirty calls per
microbatch match15 MoE layers; layer indices below count only MoE layers from0.
The hybrid model's corresponding global zero-based layer indices are recorded
in the CSV. For every rank/iteration the eight parameter calls precede MoE
layers0,1,3,5,7,9,11,13 (always before FC1).

Each entry below includes the expert layer's two ranges and their internal gap,
plus the gap from the previous expert layer's end. Layer0 excludes startup.
These are temporal allocations, not causal attribution of each gap to Scatter.

| MoE layer | F768 NEP-minus-healthy ms | F3072 NEP-minus-healthy ms |
| --- | ---: | ---: |
|0|-0.073|-0.068|
|1|1.656|7.107|
|2|0.095|0.255|
|3|1.854|7.303|
|4|0.065|0.396|
|5|1.939|7.322|
|6|0.117|0.362|
|7|1.978|7.274|
|8|0.352|0.336|
|9|2.248|7.395|
|10|0.356|0.307|
|11|2.395|7.270|
|12|0.532|0.427|
|13|2.325|7.333|
|14|0.384|0.376|

F3072: layers1,3,5,7,9,11,13 together contain51.004ms of53.396ms extra
expert-bounded first-forward span (95.5%). The first expert itself is unchanged;
startup before its first marker adds3.578ms versus healthy, separately from
the53.396ms plotted span. Thus a layer-zero-only explanation is ruled out.

Partition of53.396ms: grouped operation ranges+2.464ms, internal FC gaps+0.027ms,
inter-layer gaps+50.905ms. Subtract each configuration's ordinary-forward
pattern: the first-microbatch-specific penalty is47.821ms, of which47.754ms
is inter-layer gaps. This localizes the penalty but does not prove all47.754ms
is direct transfer service; native all-gather waits, native model-EP effects,
and uninstrumented work are not individually timed here.

The final unchanged anchor reproduces the pattern. F3072/sequence2048 also
has the same seven boundary concentrations, with6.1-6.9ms added per boundary.

## Artifacts And Validation

### Follow-up: Native All-Gather Visibility And Overlap Window

The custom `cuda_event_capture_adapter.py` records six CPU/CUDA-event wrapper
ranges, not NCCL kernels. Native parameter all-gather is not wrapped and native
record_function scopes are not recorded. Its duration/readiness is genuinely
missing, an instrumentation gap rather than evidence of zero overlap.

Code schedules native all-gather one logical bucket ahead. The first bucket
covers one MoE layer, later buckets two plus intervening model work. Actual
all-gather overlap is unmeasured. The earlier1.4ms single-expert comparison
was therefore too narrow to describe the full potential prefetch window.

Full owners, ordinary MB1/MB2, mean over measured40-49: first-FC1 to next
bucket's first-FC1 marker elapsed intervals (not pure compute/free bandwidth):

| Configuration | First single-layer interval | Six subsequent two-layer intervals |
| --- | ---: | --- |
| F3072,S8192 |6.814ms|12.637,10.948,12.222,10.782,12.228,10.894ms|
| F3072,S2048 |3.791ms|6.846,6.602,6.855,6.846,6.942,6.688ms|

Subtracting about4.5ms redistribution from these two-layer elapsed windows
gives conditional budgets of6.3-8.1ms at8k or2.1-2.4ms at2k for all-gather
and scheduling. These are NOT measured leftover slack or actual prefetch
launch-to-consumer deadlines: the intervals include model-EP and other work,
and all-gather service/participant waiting is not separately observed.
No new jobs, instrumentation or runtime changes; no claim of achievable speedup.

Analyzer: `scripts/nonuniform/analyze_nep_width_delivery.py`.
Lustre: `slurm_runs/soft_clique_baseline/3063815/width_delivery_analysis/`:
`summary.json` contains cohort means/SD/samples, per-layer ordinary and first
forward values, every parameter-call lead/stream statistic, runtime hashes;
`layer_contrasts.csv` contains all60 case/layer additive comparisons.

The existing analyzer validates event counts, clock bounds, matching CPU/GPU
labels and iteration partitions on all120 profiles. New checks validate
two-call-per-layer partition sums, parameter-call count/next-consumer mapping,
monotonic nonnegative within-forward intervals, and per-layer deltas summing
to the previously recorded first-forward gaps. Exact kernel service and native
all-gather lead time remain unmeasured; no kernel-level claim is made.
