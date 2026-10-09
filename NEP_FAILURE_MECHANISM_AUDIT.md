# Failure-mechanism audit, 2026-09-09

## Verdict

The six figures do not yet meet the goal of explaining the sources of NEP's
iteration overhead and tracing their reduction to demonstrated parity. They show
useful timing changes and residency observations, but not a verified causal
decomposition of the iteration gap. The previous completion claim was too strong.

Inputs audited: `analysis/nep_failure_mechanisms_controlled_20260909` under
`/lustre/fsw/coreai_comparch_sysarch/darfeen`, source summaries, all-rank compact
caches, raw job 2977274 NEP trace metadata, and both measurement scripts.

## Findings

### 1. Additive closure does not explain the overhead

`scan_rank()` in `scripts/nonuniform/plot_nep_failure_mechanism_details.py` clusters
CPU autograd scopes by their largest gaps. Forward/inter-microbatch time is the
remainder after subtracting other phases. Stacks close by construction. The
inspected rank-0 microbatch boundaries are well separated, but these are host
spans, not GPU compute spans or attributed GPU delays.

| Case | Logged NEP-H gap, ms | Profile-step NEP-H gap, ms |
| --- | ---: | ---: |
| a8b GA4 | 767.9 | 1101.7 |
| a8b GA20 | 819.3 | 1081.1 |
| a500 GA12/12 | 266.0 | 511.6 |
| a500 GA12/9 | 31.6 | 117.1 |

These use different windows: profiled steps 5-6 versus timing steps 8-10.
Instrumentation and boundary differences remain unreconciled. Profile components
cannot numerically explain the logged gap. The logging timer uses
`elapsed(barrier=True)`; its output alone is not an all-owner distribution of
local completion times.

For a500 the profiled gap shrinks by 394.6 ms. The change in NEP-minus-healthy CPU
phase deltas is +77.2 ms forward/inter-microbatch, -9.8 ms backward, -1.9 ms grad
tail, +0.5 ms optimizer, and -460.5 ms post-optimizer. The GPU dependencies and
host waits behind the dominant post-optimizer change have not been established.
Renaming a residual as a phase does not resolve the missing explanation.

### 2. Expert EDP mixes gradient and parameter synchronization

`classify_process_group()` in `scripts/nonuniform/analyze_workload_parity_pair.py`
classifies by group description. `edp_wait()` also accepts every EDP collective.
Both include parameter all-gather (AG) and gradient reduce-scatter (RS). Raw trace
metadata confirms these types and the following per-step kernel residency means:

| a500 case | Healthy grad RS | NEP grad RS | Healthy param AG | NEP param AG |
| --- | ---: | ---: | ---: | ---: |
| GA12/12 | 375.897 | 604.879 | 4.744 | 69.016 |
| GA12/9 | 317.297 | 1.690 | 5.853 | 19.218 |

All values are rank-0 milliseconds. Healthy issues four RS and four AG calls;
NEP issues eight of each. Thus 673.9 -> 20.9 ms is combined EDP-group residency,
not gradient-sync-only time. The after value is predominantly parameter AG.
Matching the requested bucket setting did not produce equal actual bucket counts.

### 3. Participant durations support skew, not exact transfer service

The minimum participant kernel duration is a descriptive proxy. Contention and
execution/completion behavior can differ between ranks, so it does not establish
an exact transfer-service interval. The claim of measured 2.53 ms service is withdrawn.

Matching actual group ranks, collective type, ordinal and message size gives:

| NEP case | Rank-0 gradient RS, ms | Rank-8 gradient RS, ms |
| --- | ---: | ---: |
| GA12/12 | 604.879 | 1.682 |
| GA12/9 | 1.690 | 281.960 |

This strongly supports reversal of readiness skew after decreasing reduced-replica
work. Waiting shifts to the reduced rank; it does not disappear across the job.
Healthy already has the same direction of skew as proportional NEP: rank 0 is
375.9/317.3 ms while rank 8 is about 1.25/1.22 ms. The additional NEP upstream
delay has not been fully separated from baseline asymmetry and host/dependency gaps.

For category 1, `polling_wait()` subtracts a minimum whole-step per-rank polling
total from rank 0's total. Different ranks can be late on different calls. This
is neither per-operation arrival waiting nor identification of the delaying peer;
the plot title overclaims that it identifies the source of the dominant wait.

### 4. Temporal overlap does not establish useful hiding

Union/intersection accounting correctly avoids duplicate time within a family.
However, EDP overlapping a HybridEP polling kernel can represent concurrent waits.
It does not establish model progress or saved iteration time.

a8b NEP EDP overlap rises 40.1 -> 272.5 ms. Of that 232.4 ms increase, 85.3 ms
is labeled compute and 147.0 ms other work. The compute category includes
unclassified kernels, memcpy and memset, so it is not verified useful model compute.
EDP also includes parameter AG. In a500 the model-EP exposed delta rises
181.4 -> 333.7 ms despite improved latency. The figures show that redistribution
without establishing the dependency chain that explains it.

### 5. Reliable >=95% parity is not demonstrated for category 2

Only three post-profile timing samples were used. GA12/9 healthy is
1158.5, 1169.7, 1291.9 ms; NEP is 1229.8, 1239.0, 1246.1 ms.
Mean parity is 97.45%, median parity 94.41%, and profile-span parity 93.60%.
The slow healthy sample must remain reported; its cause is unknown. More repeated
measurements are required to establish reliable >=95% parity.

Same allocation fixes the earlier GPU-generation confound but does not eliminate
run-order effects, host/runtime variance or profiling perturbation. The healthy
references retain substantial profiled skew.

### 6. The figures do not fulfill the intended presentation

Visual inspection confirms unreadably overlapping x-axis labels in the published
category-2 operation figure; category 1 uses the same crowded layout. The plots
omit timing/variance context and most workload metadata. Main panels show rank 0;
owner-range statistics are not plotted, and category 1 omits reduced-rank analysis.

## Conclusions that remain supported

- Category 1 shows amortization: five times GA increases healthy iteration time
  approximately fivefold with similar absolute NEP overhead, improving parity from
  83.93% to 96.12%. The exact upstream overhead source remains unresolved and the
  two cases use separate GB300 allocations.
- Category 2 shows a substantial latency improvement when reduced GA changes
  12 -> 9 on one allocation. Gradient RS duration asymmetry reverses between owner
  and reduced ranks, supporting work/readiness imbalance as an important mechanism.
  The exact iteration milliseconds attributable to that mechanism remain unproven.
- Small Gather/Scatter residency and correct interval-union accounting are useful
  observations; they do not close the host/dependency gaps.

## Needed to finish the original explanation

1. Split gradient RS, parameter AG, gradient reshard and parameter redistribution
   by operation and phase, recording actual bucket counts and payloads.
2. Use the existing raw traces to connect the large post-optimizer span and delayed
   model-EP to GPU completion, host waits and the peer's readiness. Apply identical
   boundaries to healthy and NEP and explicitly retain unresolved intervals.
3. Show relevant owners and peers with exclusive wall-time accounting separate
   from operation residency. Do not treat minimum-duration proxies as exact service.
4. Obtain more warmed timing samples with profiling retained, then qualify parity
   according to variance. Redraw readable panels with workload settings and separate
   profile/timing values.

This audit changes no training implementation and submits no jobs.
