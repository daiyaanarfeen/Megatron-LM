# High-Parity Compute-Window Sensitivity

User-approved priority: run the experiments most likely to change parity by
more than 2 percentage points (3+ is a strong effect). Do not spend another
full sweep establishing sub-point scheduling effects first.

Start from the previously 97.203% companion-parity a8b case: GB300, EP16/12,
TP2/CP1, H4608/latent1152/expert F3072, 512 experts/topk6, 15 MoE layers,
MBS2/GA8, sequence8192, DistOpt/HybridEP/native graphs, early redistribution,
depth1 lookahead, one delivery lane. Healthy EP16/16. Replica local batches
128/96 NEP and128/128 healthy; true GBS224 and256 respectively.

Only independent workload setting changed: sequence8192/4096/2048. Keep
parameter buckets, actual parameter bytes, expert identity, routing algorithm,
placement, optimizer, graph mode, scheduling and work-allocation ratio fixed.
Token-dispatch bytes and compute shapes necessarily change. This is a workload
sensitivity, not a perfectly isolated causal compute-only intervention.

No production implementation edits. Separate branch from11e2388c6. Only add
sequence2048 to the harness and counterbalance cases. Largest-memory8192 NEP
profile runs first. Reuse existing numerical and fail-closed training gates.
Eight GB300 nodes, segment4, each replica inside its own NVLink domain;
cross-replica RDMA verified. Do not substitute GB200 and pool results.

Three repeats, six cells, each with all-rank Nsight/custom AG/RS/Gather/delivery
profiling and a separate previously authorized timing companion:36cases.
35iterations, matched25-30 after24warmup. Orders:
- r0: N8K,H8K,N4K,H4K,N2K,H2K; profile then companion.
- r1: H2K,N2K,H4K,N4K,H8K,N8K; companion then profile.
- r2: N4K,H4K,H8K,N8K,N2K,H2K; profile then companion.

Primary parity is healthy owner iteration / NEP full-owner iteration for the
same sequence,repeat,and measurement mode. Report every repeat, mean and range;
do not count ranks or adjacent steps as independent experimental repeats.
Call >2pp consistently across repeats meaningful; retain smaller effects but
do not describe them as explaining high parity. Three repeats do not establish
population-level significance across allocations.

Measure absolute first-forward demand-to-ready waiting, native AG dependency
prefix, redistribution exposure and native compute/model-EP changes. Report
full phase excesses including final backward and optimizer/tail. A lower
parity with unchanged absolute overhead is amortization, not evidence of lost
hiding. Flag phase speedups or instrumentation-induced reversals rather than
discarding inconvenient iterations. No post-profile timing substitution.

All outputs and analysis on Lustre; final plots published to Talos. Poll
queued/running job every300s and stop on validated hangs or audit failures.

## Completed Results

Job3241762 completed0:0 in2h33m30s on eight GB300 nodes. Full replica:
theia0043,0045,0050,0053 in nvlblk19; second replica:0079,0081,0082,0090 in
nvlblk21. All36cases pass, with540complete every-rank event/Nsight/SQLite
triples. Runtime SHA256 was identical in every case. No hang or OOM.
Cross-replica RDMA, native AG/RS fingerprints, exact NEP payloads, canonical
weights, routing unit and per-topology bucket layouts all pass validation.
Largest-case NEP peak allocated135.525GiB. No production changes were needed.

Means of three repeat-level owner timings, matched steps25-30. Parity is the
mean of the three paired healthy/NEP ratios, not the ratio of pooled means.

| Sequence | Companion healthy ms | Companion NEP ms | Extra ms | Companion parity | Profiled parity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 3164.567 | 3234.416 | 69.849 | 97.8405% | 97.8518% |
| 4096 | 1702.920 | 1790.084 | 87.164 | 95.1308% | 93.9369% |
| 2048 | 1035.008 | 1129.915 | 94.907 | 91.6005% | 90.2621% |

Companion parity drops relative to8192:
- 4096:2.7097pp mean; paired repeats2.4938,2.6703,2.9649pp. All exceed2pp.
- 2048:6.2399pp mean; paired repeats5.9334,6.4117,6.3748pp. All exceed3pp.

This satisfies the user's practical effect threshold within this allocation.
Three repeats are not a statistical generalization across hardware/allocations.

## What Actually Became Less Hidden

All-owner means from the three profiled repeats:

| Sequence | Healthy later-bucket wait ms | NEP later-bucket wait ms | Last NEP bucket ready-before-demand ms |
| --- | ---: | ---: | ---: |
| 8192 | 0.026 | 0.012 | +15.368 |
| 4096 | 0.099 | 0.516 | +0.151 |
| 2048 | 3.215 | 6.448 | -5.170 |

These exclude initial-bucket waiting; negative readiness margin means late.
The final prefetched bucket changes from comfortably early to missing its
consumer deadline. At2048 the preceding bucket is also1.157ms late on average,
versus22.698ms early at8192. AG completion and redistribution GPU start are
adjacent:the existing early policy is submitting eligible work,not intentionally
waiting for the consuming layer.

Kernel-level check across all12full-replica owners, first repeat, same six
measured steps:NEP redistribution/staging union18.281/15.946/15.187ms at
8K/4K/2K, with0.388/0.981/4.571ms not overlapped by any other kernel. Residency
includes participant waiting; these are not pure wire times. The lower total
residency at2K does not negate the increased exposure or missed deadlines.
Kernel overlap is a diagnostic,not an additive causal decomposition of latency.

## Important Negative Result

This is a large parity sensitivity, but NOT evidence that lost parameter
delivery hiding caused the entire parity drop. First-forward excess remains
26.784/26.679/26.667ms at8K/4K/2K despite later bucket waits increasing. Other
first-forward components offset those waits. The shortened iteration makes
the largely fixed first-forward overhead relatively more expensive.

Profiled phase excesses,NEP minus healthy,ms:

| Sequence | First F | Other F total | Nonfinal B total | Final B | Outside F/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 | 26.784 | 1.542 | -0.288 | 23.117 | 19.412 |
| 4096 | 26.679 | 0.954 | 1.952 | 22.638 | 60.424 |
| 2048 | 26.667 | 2.180 | 4.266 | 22.177 | 59.597 |

Most of the additional profiled absolute gap localizes outside forward/backward,
not to first-forward delivery. Instrumentation is itself differential:profiled
total gaps70.576/112.649/114.881ms versus companion69.849/87.164/94.907ms.
Do not use these separate profiled phase values to allocate companion latency,
or claim the outside-F/B increase is entirely profiler overhead. The companion
absolute gap still increases17.315/25.058ms relative to8K; its complete causal
attribution is not established by this experiment.

Conclusion:shorter sequences reduce delivery slack and eventually expose later
buckets;the parity change comfortably clears the requested threshold, but much
of it is amortization, and the measured hiding loss alone does not explain it.
The five mechanisms are not thereby ranked or fully isolated. No further
scheduling/transport experiments were submitted after this test.

## Artifacts

Root:`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3241762`.
Final analysis:`sequence_hiding/summary.json`; analysis sources archived under
`analysis_source`. The three-repeat aggregate uses all owner rank timings and
all-rank event profiles; first-repeat kernel diagnostics use owners0-11 only.

Plots, tables, and this report:
https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_sequence_hiding_3241762/
