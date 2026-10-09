# Remaining a8b overhead after compact parameter transfer

## Latest completed attribution: job3090523

Completed0:0 on2026-09-17,15:25:51-15:45:24PDT. Healthy and compact NEP
each completed45iterations, with all32/28rank operation profiles for25-34.
Same allocation/resources, healthy then NEP. Both replicas fit inside their
own MNNVL clique; cross-replica transport is NET/NCCLRDMAPlugin/GDRDMA,
not NVLink. No runtime implementation changes were made for this attribution.

Workload: GB200,15MoE layers (reduced depth because20MoE OOM), H4608,
latent1152, expertF3072,512experts, top-k6, seq8192, MBS2, GA8, TP2, CP1.
Healthy EP16/16, replica batches128/128, GBS256; NEP EP16/12, batches128/96,
GBS224. DistOpt, Flex/HybridEP, selective recompute and CUDA graphs retained.
Actual expert buckets5healthy/8NEP, dense3both: not an iso-expert-bucket test.
Dynamic HybridEP sizing and random-logit force balancing are unchanged;
the latter does not guarantee identical per-expert token counts.

### Measured parity and additive attribution

Logged iteration means25-34: healthy4942.33ms (SD28.53), NEP5345.87ms
(SD34.51), gap403.54ms, **92.45% parity /8.17% extra elapsed time**.

Disjoint GPU-event-anchored phase means, full-replica owner ranks0-11:

| Phase | Healthy ms | NEP ms | Extra ms |
|---|---:|---:|---:|
| Startup plus first forward | 150.490 | 229.011 | 78.522 |
| Other seven forwards | 894.468 | 912.995 | 18.527 |
| Earlier seven backwards | 2371.755 | 2356.933 | -14.822 |
| Final backward | 510.550 | 797.611 | 287.061 |
| Backward-to-optimizer drain | 48.212 | 82.106 | 33.893 |
| Optimizer and subsequent tail | 22.795 | 32.406 | 9.611 |
| Forward/backward and microbatch transitions | 944.359 | 935.122 | -9.236 |
| Total | 4942.629 | 5346.185 | 403.556 |

Outer event anchors differ slightly from the logger; the delta differs
by0.016ms. Startup/firstF accounts for78.522ms, finalB/drain320.955ms,
everything else4.079ms net. These are additive locations of overhead,
not independent estimates of removable transfer time.

### First forward: parameter-delivery deadlines remain exposed

At eight parameter buckets, the latest full-replica redistribution finishes
3.418,6.179,8.453,10.291,8.448,8.294,8.487,8.419ms after the latest owner AG
completion. First-forward inter-expert gaps exceed the next microbatch's by
8.3-12.3ms at those delivery boundaries; healthy's corresponding penalties
are at most1.12ms. Compact transfer improved this path, but did not eliminate
AG -> packing/local redistribution -> parameter-consumer dependencies.

Owner summed AG elapsed grows44.558->53.766ms. Summed post-last-ready
AG grows40.747->53.737ms, consistent with larger logical owner payload.
These overlapping sums and cross-rank maxima MUST NOT be added to the
78.522ms phase delta. They include scheduling and are not wire service times.

### Final backward: accumulated reduced-replica lag becomes owner waiting

Using ALL healthy second-replica ranks16-31 and NEP reduced ranks16-27:

| Measurement | Healthy ms | NEP ms |
|---|---:|---:|
| Ordinary backward, full owners, microbatches1-6 | 338.660 | 336.478 |
| Ordinary backward, second/reduced replica, microbatches1-6 | 357.499 | 387.105 |
| Expert backward+recompute CPU thread time per ordinary backward, second/reduced | 82.630 | 102.997 |
| Second/reduced arrival behind full owners at final-B start | 151.011 | 374.030 |
| Second/reduced arrival behind full owners at final-B end | 12.138 | 30.592 |
| Second/reduced final-backward duration | 371.678 | 454.173 |

The exact elapsed identity is:

    owner final-B increase287.061ms
      = second/reduced final-B increase82.495ms
        + additional start skew223.019ms - additional end skew18.454ms

Thus most of the owner final-B increase is lag accumulated BEFORE final-B,
not287ms of additional RS transfer. Reduced ranks own42/43experts rather
than32. Extra native expert CPU work is measured (+20.367ms per ordinary
backward); expert ranges also grow. These profiles do not separate pure GEMM
kernel time from host-launch/dependency gaps, so poorer GEMM throughput is
NOT established as the cause of the whole29.606ms ordinary-B increase.

Owner RS summed elapsed275.728->993.141ms; before-last-participant-ready
193.692->883.714ms; afterward82.036->109.427ms. These ranges overlap each
other. The +717ms sum is not +717ms iteration overhead. Post-ready growth
is approximately33%, consistent with payload growth, but is not pure wire time.

Concrete iteration25: full owner0 submits a MoE backward graph at101.120ms
after final-B begins, spending0.122ms on the CPU call. Its GPU event range
is104.700-444.044ms. Owner0's first RS input is ready at42.043ms; reduced
rank16's corresponding input is not ready until427.662ms; RS completes at
441.933ms on owner0. The long graph is already submitted, so this particular
339ms range is not a late host launch. Its completion tracks the pending RS.
The operation-only trace does NOT resolve the internal graph dependency,
NCCL scheduling or resource contention responsible; correlation alone does
not establish which of those mechanisms stalls the graph.

### Confirmed avoidable synchronization in NEP dispatch

token_dispatcher.py:1990 stores immutable active expert slot indices on CUDA.
dispatch_postprocess, around2170, converts them with
`active_local_expert_slots.to(tokens_per_expert.device)` on every dispatch.
All-rank diagnostic metadata confirms HybridEP token counts are on CPU,
while NEP slot indices are on CUDA. Default blocking conversion therefore
copies CUDA->CPU240times/iteration, including the full replica's identity map.

| Per-iteration / per-call measurement | Healthy owners | NEP owners |
|---|---:|---:|
| Postprocess CPU elapsed sum, ms/iteration | 0.934 | 79.795 |
| Postprocess GPU-event elapsed sum, ms/iteration | 0.575 | 11.515 |
| Expert CPU start minus dispatch GPU completion, ms/call | -0.121 | +0.239 |
| Expert CPU calls beginning before dispatch GPU completion | 94.5% | 0% |

Second/reduced ranks likewise change from-0.028 to+0.322ms/call and58.7%
to0% early submission. This identifies an unnecessary synchronization and
loss of host submission lead, NOT79ms of directly additive exposed cost.
Some CPU waiting overlaps prior GPU work. Caching a CPU copy of the immutable
indices is the narrow candidate fix; it has not been applied or A/B measured.

### Measurement limits and reproducibility

- Post-profile36-45 means4821.71/5284.33ms imply91.24% parity. Measured-window
  overhead relative to post is120.62ms healthy and61.54ms NEP: this comparison
  favors NEP by59.08ms during measurement. Drift prevents treating this as an
  exact profiler correction; use92.45% only for the profiled table above.
- Healthy and NEP earlier-B phase distribution differs from job3089211 below.
  Do not mix runs or substitute its smaller final-B delta into this table.
- CUDA-event operation ranges are not individual kernel traces; unscoped
  time is not automatically idle. CPU scopes can be nested and non-additive.
- All collective participant sets matched successfully across all60ranks and
  ten steps, including AG/RS, NEP Gather and redistribution. Four synthetic
  interval tests and all60rank partitions pass. Runtime worktree is unchanged.
- Data/cache root:
  `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3090523/gb200_after_overhead_analysis/`.
  `summary.json`, `operation_partition.json`, `delivery_chain.json`, and
  `healthy-rank-*.json` / `nep-rank-*.json` preserve measurements. Analysis:
  `analyze_a8b_current_overheads.py`, `analyze_a8b_operation_partition.py`,
  and `analyze_a8b_delivery_chain.py` under scripts/nonuniform.

## Historical job3089211 evidence boundary

Matched GB200 healthy/compact-NEP job3089211, measured iterations25-34 only.
15MoE layers (depth reduced for GB200 memory), H4608, latent1152, F3072,
512experts, top-k6, sequence8192, MBS2, GA8, TP2, EP16/16 vsEP16/12.
DistOpt and Flex/HybridEP; graph and profiling settings unchanged.

Healthy4569.42ms, NEP4980.52ms: +411.10ms, 91.75% parity. Post-window timing
is4556.27/4944.34ms and is a distortion check, not a replacement measurement.
The phase-anchor partition differs from logged iteration time by about0.04ms
in the delta because its outer timing markers differ from the logger's.

These CUDA-event traces contain elapsed operation ranges, including waits,
not pure GPU kernel busy time. No gap below is automatically called GPU idle.
This section records evidence available before job3090523 completed above.

## Disjoint owner timeline

Mean over full-replica owners0-11 and the same ten measured iterations:

| Phase | Healthy ms | NEP ms | Extra ms |
|---|---:|---:|---:|
| Startup plus first forward | 140.624 | 224.839 | 84.215 |
| Other seven forwards | 815.729 | 857.464 | 41.735 |
| Earlier seven backwards | 2189.463 | 2263.513 | 74.050 |
| Final backward | 437.058 | 592.915 | 155.857 |
| Backward-to-optimizer drain | 47.755 | 72.022 | 24.267 |
| Optimizer plus subsequent tail | 22.687 | 31.697 | 9.009 |
| Forward/backward and microbatch transitions | 916.325 | 938.334 | 22.009 |
| Total additional elapsed | | | 411.142 |

These rows are additive. Collective, CPU, and expert-range measurements below
are overlapping explanations of these rows and MUST NOT be added to them.

## Supported mechanisms

### Parameter delivery still limits first forward

NEP has repeated9-12ms extra inter-expert gaps on the first forward versus
its next microbatch. Healthy's corresponding gaps are0.06-0.91ms. The gaps
cluster at parameter-bucket boundaries. After the latest owner's AG finishes,
the full replica's last parameter redistribution finishes3.53,6.30,8.55,10.37,
8.55,8.48,8.57,8.62ms later across the eight buckets. Compact transfer reduced
payload, but did not remove the dependent AG-to-local-delivery stage or its
consumption deadlines. These delays include packing, participant readiness,
copies and communication; they are not pure NVLink service times.

### Final backward includes lag accumulated before final backward

The second-replica minus full-owner mean arrival skew at final-backward start
is109.79ms healthy and204.93ms NEP. At final-backward end it is11.91/21.33ms.
Using ALL16 healthy second-replica ranks, not only ranks16-27, gives:

    extra owner final-backward span
      = extra second-replica final-backward span
        + extra start skew - extra end skew
      = 70.14 + 95.14 - 9.42 = 155.86ms

This is an elapsed-time identity, not a counterfactual estimate of removable
waiting. In particular, attributing all155.86ms to extra RS wire transfer
would be wrong.

Ordinary backward (microbatches1-6, zero-based) lasts312.67/322.93ms on
healthy/NEP owners and324.37/352.53ms on the second/reduced replica. Measured
CPU thread time inside expert backward plus recompute is72.97/72.45ms on
owners and74.48/92.40ms on the second/reduced replica. Reduced ranks have
42or43 real experts instead of32. Increased native per-expert host work is
directly measured, but how much is unhidden is not proven by adding CPU times.
The new scopes separate dispatcher, expert preparation and native graph work.

### Synchronization grows, but its longest ranges mostly contain waiting

Per-owner sums for expert RS: healthy227.89ms, NEP601.56ms. Matching all
participants' input-ready markers separates144.23/489.65ms before the latest
participant from83.66/111.91ms afterward. The latter still includes execution
and scheduling, not pure transfer. Concurrent RS ranges make the sums
non-additive; NEP's union is428.53ms rather than601.56ms.

AG after-last-ready sums41.34->54.81ms and RS83.66->111.91ms increase about
one third, consistent with the larger logical owner payload at EP12 versus
EP16. This does not prove every microsecond of that increase is wire time.
Realized expert AG/RS counts are5healthy and8NEP despite requested DDP8;
dense AG/RS counts are3in both. This is NOT an iso-expert-bucket experiment.

## Still unresolved

- Exact operation-level source of ordinary owner gaps and reduced-replica
  extra time not inside expert ranges. Existing labels omit router metadata,
  HybridEP operations, expert preparation, and native CUDA graph replay.
- Exact critical-path split of final-stage Gather/staging/RS and native work.
  Host calls and summed overlapping GPU ranges cannot answer this alone.
- The workload intentionally retains dynamic HybridEP receive sizing
  (capacity factorNone). Static-budget mode was removed from the original
  matrix after overflow failures (journal2026-08-25). Do not silently switch
  this setting while attributing the current measured gap.

Analysis: scripts/nonuniform/analyze_a8b_operation_partition.py, validated on
four synthetic interval cases and all60existing rank caches. Its disjoint
scope partition explicitly labels unmeasured time as unscoped, never idle.
Raw/cache data remain under Lustre soft_clique_baseline/3089211.
