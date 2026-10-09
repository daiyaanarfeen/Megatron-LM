# First-forward overhead with the accepted changes

## Measurement

Existing GB200 job3122515 only; no new GPU jobs or training-code changes. Compare
`cleanup_ab_healthy_margin` against `cleanup_ab_cleanup_margin`: both use native
backward SM margin32; NEP also uses buffer-cleanup commitee28f09ff. Rejected parallel
owner transfers are not enabled. Trusted early parameter-submission behavior is retained.

Hard a8b latent workload: 15 MoE layers in the hybrid/MTP pattern, H4608/F3072,
latent1152, E512/top-k6, sequence8192, MBS1/GA2, TP2/CP1, Flex/DistOpt/native graphs.
Healthy EP16/16 local batches16/16, NEP EP16/12 local batches16/12. Each replica is
inside one physical NVL domain; domains12/16 are separate. Native requested DDP buckets8;
actual expert buckets healthy5 versus NEP8, unchanged from this accepted comparison.

Matched measured iterations20-27 after19 warmups. All full-replica owners0-11 are
averaged within each iteration. First forward: healthy99.842761 ms, NEP137.452227 ms,
additional37.609467 ms. These are profiled, not post-profile, timings. The existing
instrumentation-distortion caveat remains; no timings from another window are substituted.

## Additive accounting

Partition each rank's first-forward timeline into captured-layer intervals and gaps.
Inside a captured layer, classify native graph-kernel occupancy; asynchronous parameter
work overlapping it is not added again. Outside layers, partition all GPU activity,
retaining mixed categories. Differences below are NEP minus healthy and sum to37.609467 ms.
This locates elapsed overhead; it is not a counterfactual estimate of savings if an
individual collective were removed. Cross-rank concurrency is not summed: the table
averages the same disjoint rank-local partitions over owners.

| Observed interval | Additional ms |
|---|---:|
| Native model-EP residency inside layers | 20.259 |
| Outside-layer NEP broadcast, before last participant input-ready | 8.457 |
| Outside-layer NEP broadcast, after all participants input-ready | 1.344 |
| Outside-layer NEP GPU packing/unpacking, with no other kernel active | 2.721 |
| Native compute/TP residency inside layers | 4.048 |
| AG-only intervals outside layers, including dense/expert AG overlap | -0.163 |
| Other measured disjoint intervals: gaps, memory, mixed activity | 0.944 |
| Total | 37.609 |

The last row before total is computed from the remaining observed interval categories,
not inserted as an unexplained balancing residual. Full categories are in the JSON.
Normalized native process-group aliases are required: healthy uses long names such as
TENSOR_MODEL_PARALLEL_GROUP while NEP uses `tp`. The final extractor schema_version2
handles both. Earlier exploratory JSON files without this version are not the final
accounting; use `first_forward_findings.json`.

### Peer waiting, not slower model-EP transport

For corresponding model-EP kernels, match all16 full-replica ranks by iteration,
layer ordinal, kernel name and occurrence. Summed ag_nvl residency rises3.531 ->20.869 ms;
its prefix before the last peer's corresponding kernel starts rises2.712 ->20.064 ms.
Thus17.352 ms of the17.338 ms increase coincides with late peer start, while the
post-last-start portion is essentially unchanged. Device-sync residency rises0.559 ->3.118 ms,
with before-last-peer-start prefix0.318 ->2.770 ms. Approximately19.8 ms of the20.3 ms
model-EP increase is accounted for by these arrival prefixes. These diagnostics are
contained in the table's model-EP row, not additive extra costs. Kernel start is a
readiness proxy, not an observation of an internal NVSHMEM flag or pure wire service.

Owner broadcast kernel residency totals24.458 ms, of which21.700 ms precedes the last
participant's input-ready marker and2.758 ms follows it. Much overlaps other work.
Restricting to broadcast-only intervals outside captured layers gives8.457 ms before
readiness and1.344 ms afterward, as in the table. The latter still includes NCCL
processing/contention and cannot be called a physical transfer-time lower bound.

### Exact readiness chain example

Iteration20, second parameter bucket, times relative to rank0's first MoE start:

- Rank0 finishes this bucket's redistribution at8.837 ms, starts its consuming MoE
  at8.850 ms, and reaches ag_nvl at approximately10.699 ms.
- Rank11's AG completes9.954 ms, but its redistribution finishes14.547 ms.
  Its broadcast lasts3.855 ms;3.583 ms precedes follower15 being ready, leaving0.272 ms
  after readiness. Follower15 receives owners8,9,10,11 sequentially for this bucket.
- Rank11 then performs normal layer preparation and reaches ag_nvl at15.835 ms.
  Rank0's ag_nvl prefix waiting for the last peer is5.137 ms.

The owner-broadcast wait and rank0's model-EP wait describe a propagating readiness
chain. Their individual example durations must not be summed across ranks as an
iteration penalty. In all64 bucket/iteration instances, rank11 is last to finish
parameter redistribution. Full-replica parameter-ready spread averages6.320 ms,
range2.856-8.033 ms. Some model-EP waiting occurs after that peer's parameters are ready
because its ordinary router/pre-dispatch computation still has to run.

## Host submission versus device scheduling

All768 owner/bucket redistribution calls finish CPU submission before the matching AG
completes: minimum lead3.174 ms, mean32.962 ms. Sum of positive AG-complete-to-redistribution-
start gaps averages only0.024 ms per owner/iteration. The old late host submission issue
is not the explanation for these measurements. CPU redistribution scopes total10.106 ms,
but that host work is already submitted ahead of the GPU and is not10 ms of exposed delay.

For all672 next-bucket AG calls, CPU submission also finishes before the preceding
bucket's redistribution completes, by at least3.590 ms. Yet its GPU input-ready marker
never precedes that completion. The accepted code explains this: `finish_param_sync`
first inserts the current bucket's parameter-ready wait on the compute stream, then
calls `next_group.start_param_sync`. The next AG inherits that stream dependency.
The transfer stream also serializes owner broadcasts and bounded staging-buffer reuse.
This is distinct from the rejected experiment that parallelized owner broadcasts.

## What this establishes and does not establish

- Direct exposed packing/unpacking is2.721 ms. Direct exposed broadcast time after
  all inputs are ready is1.344 ms; most direct broadcast residency is participant waiting.
- Readiness skew dominates. Roughly19.8 ms of extra native model-EP residency plus8.5 ms
  of directly exposed broadcast waiting occur in different rank-local time intervals.
- Native compute/TP contributes4.048 ms (compute2.163, TP1.886). These profiles do not
  uniquely distinguish SM, memory-system, and link contention within that increase.
- The negative AG-only delta does not prove AG has no causal cost: its timing can delay
  redistribution and shift waiting into model-EP. Expert AG total residency actually rises
 40.704 ->56.229 ms; per-owner output payload rises6.795 ->9.060 GB (decimal), with5 ->8 calls.
- No physical lower bound or guaranteed recoverable milliseconds has been established.
  The observed37.6 ms is not simply unavoidable wire time. Testing a less restrictive
  next-AG dependency while preserving AG-before-redistribution and parameter readiness
  would be a separate implementation experiment, not a conclusion implied by this trace.

CUDA-event clock calibration bound is at most8.416 us; the largest Nsight alignment
half-interval is30.680 us. Cross-rank readiness splits are approximate at this scale;
reported millisecond differences are much larger. No fabric utilization counters or
internal HybridEP flag timestamps were collected. Parameter values were not changed.

## Reproduction

Run `python3 scripts/nonuniform/analyze_nep_first_forward_delivery.py`, then
`python3 scripts/nonuniform/summarize_nep_first_forward_delivery.py`.
Both only read saved profiles and write offline analysis under:

`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3122515/first_forward_attribution/`

Canonical output: `first_forward_findings.json`; `delivery-*-rank-*.json` retain per-rank
kernel intervals, graph partitions, AG/redistribution completion and CPU submission.
Assertions verify phase closure, peer membership, collective/kernel counts, all measured
iterations, graph ordering, and independent broadcast-prefix closure.
