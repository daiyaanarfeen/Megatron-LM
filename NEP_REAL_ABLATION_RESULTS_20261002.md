# Actual a8b Placement And Work-Allocation Ablations

## Scope And Protocol

Completed placement job3215521 (28 cases, 2h27m35s) and MBS2 allocation
job3215822 (16 cases, 1h26m15s), both exit0:0. Production Megatron code was
unchanged. These are a8b results, not a cross-model replication of all workloads.

Use the previously saved memory-limited a8b latent depth, not the four-layer
toy and not the original full-depth120B model: 15 MoE layers in the hybrid/MTP
pattern MEMEMEM*EMEMEMEMEM*EMEMEMEME/*E/*E; H4608, latent1152, expertF3072,
512 experts, top-k6, sharedF6144, vocabulary131072, S8192. TP2, CP1, ETP1,
PP1, BF16 distributed optimizer, Flex/HybridEP and native CUDA graphs for
Mamba, attention and MoE. Healthy EP16/16; NEP EP16/12. Both use MBS2.

Each experiment has two reversed-order repeats, 35 iterations per case, and
matched measurements of steps25-30 after24 warmup iterations. Every profiled
case has a separate approved timing companion. Neither post-profile windows
nor correction/subtraction of profiler overhead are used. Iteration timing
is the mean of the native interval timer on owners0-11. Parity is healthy
owner time / NEP owner time, averaged over matched repeats. This is latency
parity, not equal total token throughput or a convergence comparison.

All ranks have Nsight and operation-range records for parameter AG, gradient
RS, NEP Gather and parameter redistribution. Operation ranges can contain
dependencies/waits and are not pure transfer or compute service times.

Both studies ran on GB300 with eight nodes and segment4. Placement used full
replica nvlblk18 and second replica nvlblk27; work allocation used nodes
theia0005/0006/0007/0010 and theia0278/0280/0281/0284. Each replica remains
within one NVLink domain. Transport audits pass: cross-replica pairs use RDMA,
intra-replica communication uses P2P. Canonical expert initialization,
randomized exact-balanced routing, transfer split agreement, fixed useful
transfer bytes across placement policies, and native optimizer AG/RS payload
checks pass. TP AG/RS counts legitimately change with microbatch count and
are not incorrectly treated as optimizer payload changes.

Requested DDP buckets8 produces dense3/3 and expert5/3 actual buckets
(healthy/NEP), fixed within each topology throughout this study. This is not
an iso-bucket comparison. The current runtime/routing controls are not an
exact replay of the historical result: a fresh high-parity control was
established before the ablations. Both transfer buffer lanes are preallocated
for every policy; only the number of active streams changes.

## Placement And Transfer Scheduling

All cases have GA8/8: healthy local batches128/128, trueGBS256; NEP128/96,
trueGBS224. Backward owner order is fixed ascending. The changed transfer
order applies to forward parameter delivery. Disjoint-first prioritizes
transfers whose participant groups do not intersect.

| Policy | Companion owner ms | Companion parity | Profiled owner ms | Profiled parity |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 3159.68 | 100.00% | 3224.83 | 100.00% |
| Compact, ascending, 1 stream | 3240.00 | 97.52% | 3296.46 | 97.83% |
| Compact, disjoint-first, 1 stream | 3239.95 | 97.52% | 3298.67 | 97.76% |
| Compact, ascending, 2 streams | 3264.70 | 96.78% | 3323.15 | 97.04% |
| Compact, disjoint-first, 2 streams | 3264.40 | 96.79% | 3322.12 | 97.07% |
| Spread, ascending, 1 stream | 3256.15 | 97.04% | 3395.50 | 94.97% |
| Spread, ascending, 2 streams | 3245.95 | 97.34% | 3396.16 | 94.96% |

Compact single-stream policies tie: their mean difference is0.05ms, and
their ordering reverses across repeats. Two compact streams add about24ms
(0.75%) rather than improving latency. Spread/two-stream is slightly better
than spread/one-stream in these companions, but neither beats compact/one.
All companion policies retain96.78-97.52% parity: the large toy-workload
placement sensitivity does not carry over at the same relative magnitude.

Instrumentation is policy-dependent. Healthy adds65.15ms, compact policies
about56-59ms, and spread policies139-150ms. Therefore the larger spread penalty
in the profiles is not presented as its normal-run penalty. Traces and timing
companions remain separately reported. Two repeats establish these observed
differences, not universal rankings across models or allocations.

## Reduced-Replica Work Allocation

All use compact/disjoint-first/one-stream. Full replica GA8 and local128 stay
fixed. Reduced GA6/8/10 gives local72/96/120, respectively. Healthy GA8/8,
local128/128. Throughput uses true global batch, not the surrogate batch
printed by a rank-local microbatch calculator.

| Case | Full/reduced GA | TrueGBS | Companion owner ms | Companion parity | Thousand tokens/s | Profiled parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Healthy | 8/8 | 256 | 3149.11 | 100.00% | 665.95 | 100.00% |
| Proportional | 8/8 | 224 | 3231.95 | 97.44% | 567.77 | 97.70% |
| Underloaded | 8/6 | 200 | 3228.40 | 97.54% | 507.50 | 97.79% |
| Overloaded | 8/10 | 248 | 3956.85 | 79.59% | 513.44 | 78.80% |

Underloading reduces latency by3.55ms (0.11%) but loses10.62% throughput.
Overloading increases latency by724.90ms (22.43%) and loses9.57% throughput
despite processing10.71% more tokens per iteration. Proportional is the best
tested point for throughput, unlike the earlier toy overload result.

Measured profiled owner phase means, in ms:

| Case | First F | Other F | Nonfinal B | Final B | Outside F/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy | 175.03 | 1130.31 | 1610.33 | 241.64 | 51.65 |
| Proportional | 204.44 | 1131.94 | 1606.54 | 274.41 | 67.01 |
| Underloaded | 204.06 | 1130.66 | 1606.42 | 273.73 | 66.69 |
| Overloaded | 202.81 | 1131.56 | 1607.01 | 1063.17 | 67.75 |

Underloading barely changes any owner phase. Overloading adds788.76ms to
owner final backward relative to proportional, while earlier owner phases
remain nearly unchanged. Reduced-replica other-forward/nonfinal-backward
grow from1128.37/1620.96ms to1451.97/2083.99ms. This supports extra reduced
work arriving late at synchronization, not slower ordinary owner compute.
These profiled phase ranges are elapsed spans, not pure kernels, and must not
be subtracted from the companion latency gap as an exact decomposition.

## Failed Rebatching Case

MBS1/GA16 proportional caseS in job3215526 completed25 iterations and then
timed out in iteration26 during active profiling. The first watchdogs name
full-replica TP all-gathers; reduced-rank RS failures follow. No OOM evidence.
Root cause remains unresolved, and profiling is not established as the cause.
No accepted parity exists forS, and matched healthyH1 was not run. No runtime
fix was made. The completed MBS2 allocation sweep does not answer the MBS1
rebatching question or certify that configuration as working.

## Artifacts

Lustre roots: slurm_runs/nep_cleanup_ab/3215521 and3215822 under
/lustre/fsw/coreai_comparch_sysarch/darfeen. Each has actual-summary.json,
work-controls-audit.json, per-case actual-validation.json, all-rank native
timings/profiles/manifests and analysis_source_20261002 with source hashes.
Keep job3215526 for the failed case and jobs3215136/3215340 for gate history.

Plots and CSV/JSON:
https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_actual_a8b_ablations_20261002/

Offline analysis scripts executed successfully; all SVG files parse as XML.
Host import-format check could not run because uv/isort are unavailable on
the login node. No production implementation files were edited.
