# Delivery And Rebatching Ablations

Preserve the prior tested branch. This branch is based on0c8ab3e39 plus the
actual-workload harness used in3215521/3215822. Only production file changed:
megatron/core/distributed/nonuniform_ep.py. Native AG/RS code is unchanged.

## Delivery

Actual a8b memory-limited15-MoE hybrid/MTP configuration,H4608/latent1152/F3072,
E512/topk6,MBS2/GA8,EP16/12,TP2/CP1,DistOpt,HybridEP and native graphs.
At S4096 and8192 compare healthy, NEP consumer redistribution, current early
redistribution and early redistribution with two-bucket lookahead.
Local batches128/128 healthy,128/96 NEP; GBS256/224. Identical placement,
useful transfer bytes, optimizer buckets and model dimensions across policies.
Sequence length changes tokens/iteration by design; match healthy per length.

Consumer and early share native AG launch points. They differ only in when
the same dependency/redistribution function is submitted. Lookahead2 launches
the next two expert buckets in canonical order instead of one. The existing
shared delivery stream orders bounded staging-buffer reuse, so additional
transfer buffers are not presumed necessary; measure actual peak memory.
No new host synchronization, polling thread, barrier or native optimizer edit.

Profile-only markers delimit parameter demand and compute-stream readiness,
keyed by expert bucket. Existing all-rank AG/RS/Gather/redistribution, PG,
payload, host-submission and GPU-completion records remain enabled. Bucket
manifests include layer names. Event ranges are not pure transfer service time.

## Batching

At S8192,current early delivery,compare MBS/GA1/16,2/8,4/4 with matched healthy
for each. MBS*GA16 and per-replica local/global batches stay fixed. Routing is
the same repeated exact-balanced unit. This changes tokens per dispatch and
number of dispatches,not total tokens/work allocation. Compare all-forward and
all-backward sums as well as first/other-forward and final/nonfinal-backward:
different GA counts must not be mistaken for phase-efficiency changes.

## Gates

First use real native AG/RS and numerical owner/follower redistribution checks
on four GPUs,all three policies,including force-sync and repeated iterations.
Then EP8/6 four-layer training with every policy and profile validation. Only
after those pass move to EP16/12 actual workload. GB200/GB300 eligible for the
small gate. Use GB300 for actual workload because earlier reserved-memory
measurements fail the conservative GB200 headroom gate.

Memory/failure gate first: MBS4 profiled,then MBS1 companion and profiled.
The previous MBS1 timeout remains unresolved; stop on recurring failure,
preserve evidence,and consult before changing anything outside this ablation.
Do not silently remove MBS1 or replace it with a different model.

Full sweeps use35iterations,matched25-30 after24warmup,all-rank profiles and
separate approved timing companions,two reversed-order repeats. No post-window
substitution. Each replica stays within one NVLink domain; distinct clique IDs
and recorded RDMA cross-replica transport. Canonical initialization,routing,
exact payloads,memory and native optimizer manifests remain acceptance gates.
Poll Slurm every300seconds. All output goes to Lustre and completed plots to
the requested Talos directory. No promise of a favorable ablation result.

## Gate Results And Stop Point

Small job3217536 completed0:0 in14m13s onGB200. All three4GPU numerical modes
and all three EP8/6 training/profile/transfer gates passed. Container isort,
Black and compilation succeeded. The previous tested branch remains untouched.

Actual gate3217613 onGB300 passed profiledM4,P8K,H4 and the M1 timing companion.
M4 peak allocated172.115GiB,reserved216.953GiB,min end-free52.166GiB. M1 timing
companion completed35iterations. Its profiled counterpart again stopped after
iteration25 and timed out in26,matching prior job3215526. Full-replica TP
all-gathers timed out at180s; stacks also show CUDA-event recording during
graph replay. No explicit event-pool exhaustion or OOM. The evidence does not
yet distinguish profiler-caused blocking from an ordering bug exposed by it.
Canceled3217613 after watchdog/stack evidence was saved,elapsed38m43s. No full
sweeps submitted. Requested approval for Nsight-only versus event-only MBS1
profiling isolation with the training implementation unchanged.

An independent benchmark labeling error was found: healthy uses uppercase
native PG descriptors,whereas NEP uses ep_dp/dp_cp. Records were captured,but
the old healthy optimizer fingerprint was empty and H4 deadline markers were
mislabeled dense. Corrected the offline audit and revalidated all44cases in
3215521/3215822 successfully; their timings did not change. Corrected this
branch's marker classifier and future validator after the job stopped. The
saved gateH4 deadline labels remain invalid for expert-specific attribution;
do not rewrite them or use them as valid expert-deadline results. Fresh healthy
profiles are required. These label-only corrections have syntax checks but
have not yet been exercised in a new GPU job.

## October 5: MBS1 Hang Isolated To Profiling Configuration

Compared3215521/B (working MBS2/GA8) with3215526/S (hanging MBS1/GA16).
Both used identical nonuniform_ep.py SHA25620feecf1025a36a... and the same
relevant NCCL/profiling controls. The first failure therefore predates the
delivery-policy changes. Both failed MBS1 cases stop in iteration26 after25
completed; host stacks include CUDA-event record during graph replay and
native TP all-gather. The unprofiled MBS1 companion completed35 iterations.

The failed logs contain an explicit Nsight warning that device-side event
tracing may introduce false dependencies across CUDA streams. Changed only
`--cuda-event-trace=true` to `false`, retaining custom CUDA timing events,
collective observer streams, graph-node/kernel tracing, and NVTX. No NEP,
model, batch, routing, or native collective changes.

Job3238472 completed0:0 in11m31s on GB3008nodes/segment4. All28 ranks completed
35iterations, and all28 Nsight reports/SQLite exports and CUDA-event JSONs
exist. WORK_VALIDATION_PASS and ACTUAL_ABLATION_PASS include required native
AG/RS, NEP Gather/parameter transfer, routing and memory checks. Numerical
prefetch tests passed too. No NCCL timeout, OOM, NaN or skipped iteration.

Full-owner measured25-30 mean3515.274ms, sampleSD4.271ms. Pre20-23 andpost32-35
both average3430.387ms: instrumentation still adds about2.47%. No matched
healthy MBS1 run here, so this is a hang/profile-validity gate, not a new
parity result. Allocation differed from the failures; exact driver-level
dependency cycle was not recovered. The one-option successful intervention
and the documented mechanism strongly implicate Nsight event tracing, without
proving an exact internal wait graph or universal hang freedom.

Launcher default is nowfalse. All other instrumentation remains enabled.
Unused Nsight-only diagnostic plumbing was removed. Its initial job3238449
was canceled during container setup before any training case (1m46s allocated).
The old branch remains untouched, and full ablation sweeps remain unsubmitted.

Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3238472`.
Shared profiler source snapshots are in `benchmark_source/scripts/nonuniform`.
Reference: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-event-trace

## October 5: Resume Approved Sweeps

Use two GB3008-node/segment4 allocations with the validated event-trace=false
setting for all newly profiled healthy and NEP cases. Do not mix old event-trace
results into the new comparisons. All cases retain35iterations and matched
25-30 measurements, with separate authorized timing companions.

- 4K delivery: cells P4K,H4K,N4K,D4K in repeat0 and reversed in repeat1.
- 8K delivery plus batching: cells M4,H4,M1,H1,N8K,H8K,D8K,P8K in repeat0
  and reversed in repeat1. N8K/H8K are also the MBS2/GA8 batching anchor;
  no duplicate M2/H2 launches are needed.
- Repeat0 runs profile then timing companion per cell; repeat1 reverses both
  cell order and profile/companion order. Largest MBS is first at8K.
- 16cases at4K and32cases at8K, including both profiling modes and repeats.
  Every case must pass the existing fail-closed audit before the next starts.
- Production implementation remains3564f1a5c. Only launcher event-trace
  default and experiment documentation change after the completed hang gate.

## Completed Results

Both jobs completed0:0:3238625 in1h11m11s and3238626 in2h42m38s. All48cases
passed their fail-closed audits. All24profiled cases have complete every-rank
CUDA-event JSON,Nsight report and SQLite artifacts (704rank profiles total).
Cross-case canonical initialization, routing, native optimizer payload, exact
NEP transfer payload, memory, deadline and RDMA/P2P transport audits pass.
No hang or OOM recurred. Checkpoint059bbe151 was used throughout; production
code was unchanged. Monitoring was interrupted near the end of the8K sweep;
Slurm continued independently and completed the remaining cases successfully.

Means below average the two reversed-order repeat means. Parity is the mean
of matched healthy/full-owner NEP iteration-time ratios, not aggregate token
throughput. Both modes use the same iteration numbers25-30; all cases train
35iterations. Timing companions were explicitly approved and are distinct
runs, not post-profile substitutions. Each profile uses the validated fix.

### Delivery Policies

MBS2/GA8 with fixed proportional local batches128/96 for NEP,128/128 healthy.
On-demand defers parameter redistribution until consumer demand; early is
the existing implementation; deep uses early redistribution and two-bucket
native AG lookahead. Useful payload and storage layout are fixed.

| Sequence | Policy | Companion healthy / NEP ms | Companion parity | Profiled parity |
| --- | --- | ---: | ---: | ---: |
| 4096 | On-demand | 1709.591 / 1804.531 | 94.739% | 93.900% |
| 4096 | Early | 1709.591 / 1798.645 | 95.049% | 94.178% |
| 4096 | Deep | 1709.591 / 1800.614 | 94.945% | 94.472% |
| 8192 | On-demand | 3139.920 / 3231.179 | 97.176% | 97.319% |
| 8192 | Early | 3139.920 / 3230.261 | 97.203% | 97.433% |
| 8192 | Deep | 3139.920 / 3224.312 | 97.383% | 97.534% |

Early saves5.886ms over on-demand at4K, consistently across repeats. Its8K
whole-iteration advantage is only0.919ms and reverses sign across repeats.
Deep versus early has no reliable4K benefit (1.968ms slower on average,
opposite signs across repeats). At8K deep saves5.949ms (5.001/6.897ms per
repeat), only0.184% of NEP latency. This is not a large policy-sensitive regime.

| Sequence | Profile quantity, ms | On-demand | Early | Deep |
| --- | --- | ---: | ---: | ---: |
| 4096 | Owner demand-to-ready sum | 32.029 | 23.006 | 22.480 |
| 4096 | First-forward excess versus healthy | 30.715 | 26.223 | 25.705 |
| 8192 | Owner demand-to-ready sum | 31.523 | 22.308 | 22.091 |
| 8192 | First-forward excess versus healthy | 31.402 | 27.398 | 25.726 |

These are elapsed compute-stream dependency spans, not pure transfer service
times or additive causal costs. Early delivery reduces observed consumer
waiting and first-forward overhead; deep adds little further readiness
improvement. The changes do not remove the initial parameter-delivery cost.

### Fixed-Work Batching

Sequence8192,current early delivery. MBS*GA stays16, so total local batches,
global batches and tokens per iteration do not change. N8K/H8K are the
MBS2/GA8 anchor reused from the delivery sweep.

| MBS / GA | Companion healthy ms | Companion NEP ms | Extra ms | Companion parity | Profiled parity | NEP peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 16 | 3291.483 | 3421.956 | 130.472 | 96.187% | 96.114% | 127.732 |
| 2 / 8 | 3139.920 | 3230.261 | 90.340 | 97.203% | 97.433% | 135.525 |
| 4 / 4 | 2980.601 | 3063.546 | 82.945 | 97.293% | 97.780% | 172.115 |

MBS1->2 reduces NEP extra latency by40.132ms and improves parity1.016points,
at fixed total work. MBS2->4 reduces absolute NEP latency166.715ms, but healthy
also improves159.319ms; parity gains only0.090points and the direction is not
consistent between repeats. Larger microbatches primarily improve absolute
throughput beyondMBS2, with substantially higher memory demand.

| Profiled NEP-minus-healthy phase ms | MBS1 | MBS2 | MBS4 |
| --- | ---: | ---: | ---: |
| First forward | 26.415 | 27.398 | 27.265 |
| All other forwards | 4.437 | 2.386 | 3.347 |
| All nonfinal backwards | 2.893 | -0.917 | 4.381 |
| Final backward | 41.384 | 30.347 | 29.391 |
| Outside forward/backward | 61.372 | 24.859 | 4.083 |

Within these profiles, the reduction is concentrated in final backward and
outside-F/B, not first forward or ordinary microbatches. This is a phase
localization, not a kernel-level proof of why the companion timing improves.
Do not label the outside-F/B residual CPU overhead without further attribution.
Do not allocate companion timing deltas using these separate-run phase values.
The phase spans and native timers also have slightly different boundaries.

### Limits And Artifacts

Two repeats establish only within-allocation repeatability, not broad hardware
or workload generality. Healthy/NEP expert bucket counts remain5/3 (dense3/3),
fixed across the interventions but not an iso-bucket topology comparison.
The actual workload is the saved memory-limited15-MoE hybrid/MTP a8b, not a
new full-depth120B claim. The two sequence lengths used different allocations;
compare each policy to its own same-allocation healthy/control runs.

Profiling is still intrusive: profiled-versus-companion parity differs by up
to0.871percentage points (4K early), and8K early profiled parity spans
97.087-97.778% between repeats while companion parity spans97.173-97.234%.
The MBS1 companion/profile parity gap is only0.073points on average. Retain
both modes; do not claim profiling distortion is eliminated for every case.

Each job root under `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab`
contains `actual-summary.json`, `delivery-batching-summary.json`,
`work-controls-audit.json`, per-case validations and raw every-rank profiles.
Reproduce aggregation with `scripts/nonuniform/analyze_nep_delivery_batching.py <job>`.
