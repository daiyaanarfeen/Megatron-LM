# Early post-AG parameter redistribution

## Implementation

Candidate `nep/param-prefetch-20260915`, commit `51fdbbb11`, worktree
`.worktrees/nep-param-prefetch-20260915`. Trusted `nep/sync-cpu-20260914`
at `93d313df4d55` is unchanged.

Only three runtime methods in `nonuniform_ep.py` changed: `start_param_sync`,
`finish_param_sync`, and DistOpt buffer setup. Asynchronous start now queues
native AG completion and existing pack/broadcast/unpack on one shared transfer
stream. A per-bucket event joins that stream at parameter consumption.
Native AG, redistribution math, bounded two-slot storage, next-bucket prefetch,
force-sync/non-overlap behavior, and backward synchronization are unchanged.
There is no host polling thread.

## Controlled experiment

Job **3066391**, COMPLETED 0:0, GB200 `lyris[0019-0022]`, segment4,17:37
allocation time. Candidate correctness validation, healthy, old NEP, new NEP
ran sequentially on the same allocation/container.

- a770-family hybrid: 15 MoE,15 Mamba,3 GQA layers; H1280, expert F768,
  shared F1536,512 experts,top-k6,sequence2048,MBS2,GA3.
- Healthy EP8/8 versus NEP EP8/6; TP/CP/PP/ETP all1. Replica batch48/48
  versus48/36, global batch96 versus84. Flex/HybridEP, DistOpt, and existing
  local CUDA graphs retained. Native2dense/7expert healthy buckets versus
  8logical NEP expert groups, unchanged from the workload being optimized.
- Software NVLink cliques1000/1001 preserve cross-replica RDMA. NCCL logs
  confirm RDMA, not cross-replica NVLink. Old/new NEP options differ only in
  output paths; runtime hashes differ as intended.
- 60 iterations each; all-rank CUDA-event operation profiles. Both timing and
  analysis use **iterations40-49**, matched full-owner ranks0-5. Profiling unchanged.

Means +/- SD over10 iteration means across ranks, not60 independent samples:

| Metric | Healthy | Delayed NEP | Prefetch NEP |
|---|---:|---:|---:|
| Owner iteration, ms | 660.70 +/-3.82 | 900.15 +/-9.25 | 894.09 +/-8.16 |
| Owner parity | 100% | 73.40% | 73.90% |
| First-forward expert envelope, ms | 86.08 +/-0.93 | 140.39 +/-3.60 | 116.42 +/-4.36 |
| Excess first-forward envelope, ms | 0 | 54.31 | 30.33 |
| Iteration start to last expert of first forward, ms | 96.68 | 153.29 | 131.92 |

The first-forward envelope is first expert GPU-range start through last expert
GPU-range end, not a full-forward timer. It includes intervening work and
dependencies. The startup-inclusive row prevents moving work before the first
expert from being mistaken for a speedup.

First-forward envelope improves23.98ms, removing44.2% of its excess over healthy.
Startup grows2.61ms, leaving a **21.37ms startup-inclusive gain**. End-to-end
improves only6.06ms (0.67%), or0.50 parity percentage points. That gain is small
relative to iteration variation; this single sequential comparison does not
establish a robust end-to-end speedup or no regression.

## Earlier submission verified

- Across owners0-5/all10iterations, middle buckets2-7 AG-completion to
  redistribution-start interval changes9.897ms to approximately0ms. Measured
  new value-0.0073ms reflects AG completion observer recording after the actual
  dependency, not redistribution reading incomplete AG data.
- Redistribution is on a different stream from expert compute in every
  sample, versus the consumer stream previously.
- Its overlap with recorded first-forward expert operation ranges changes
  0 to9.050ms/iteration. These are elapsed operation envelopes, not proof of
  kernel-level compute/NCCL concurrency or complete GPU coverage.
- Redistribution completes before the end of first-forward experts on every
  rank/iteration; none remains queued into backward.
- Per-rank/per-iteration operation counts, collective payloads, dtypes, and
  memberships match old/new NEP on all14ranks.

## Why the end-to-end gain is smaller

Additive owner temporal-phase changes, new minus old (not causal attribution):

| Phase | Change, ms |
|---|---:|
| Before first expert | +2.61 |
| First-forward envelope | -23.98 |
| Later forwards, total | +2.76 |
| Forward-to-backward intervals | -0.11 |
| Earlier backwards, total | -2.93 |
| Final backward envelope | +10.33 |
| Between microbatches | -2.13 |
| Last expert backward to optimizer | +8.31 |
| Optimizer | -0.10 |
| After optimizer | -0.83 |
| **Entire iteration** | **-6.06** |

Full owners enter final backward23.78ms earlier and finish its expert envelope
13.44ms earlier; reduced owners finish their final expert envelope only6.34ms
earlier. The full-owner tail grows even though absolute optimizer start advances
5.14ms. Reduced-replica startup-inclusive first forward improves14.29ms, less
than full owners'21.37ms.

Longer backward/tail intervals cannot simply be called redistribution kernels
extending into backward. However, this experiment does not prove all changes
outside first forward are random: full-owner gradient-staging CPU elapsed time
rises38.70 to48.74ms while its GPU range falls15.77 to15.13ms. Cause is not
established here. Candidate remains isolated; no blanket regression-free claim.

## Correctness and limits

- Four-GPU smoke3066319: real native AG/RS, model-EP A2A and owner/follower
  broadcasts; exact fresh parameter values over5iterations; repeated finish,
  force-sync, two-slot reuse, independent compute while AG pending all PASS.
- Four focused lifecycle/transfer pytest tests PASS; isort/Black/diff check PASS.
- Workload validation: all14ranks complete8iterations with exact-output checks
  of redistribution and owner gradient staging against trusted methods on
  the same live inputs. All timing cases finish without skips/NaNs.
- These checks do not claim bitwise equality of separate stochastic training runs.
- Prior differential healthy/NEP profiling distortion is not eliminated or
  subtracted. Pre/post windows are diagnostic, not substitutes for measured parity.
- Logged pre/recorded/post parity: old71.71/73.40/72.91%; new72.30/73.89/73.98%.

## Artifacts

Lustre root:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3066391/`.
Each `healthy`, `delayed`, `prefetch` has all-rank profiles under
`cupti/ep8/scaling_ladder__a770m_4b_moe_215b/<healthy-or-nep>/3066391/cupti_profile/`.
Analysis: `prefetch_analysis/summary.json`.

Reproduce: `python3 scripts/nonuniform/analyze_nep_param_prefetch.py 3066391`.
Launch: `scripts/nonuniform/run_nep_param_prefetch_ab.sh`.

## 2026-09-16: Final-backward and tail participant-readiness audit

Existing job3066391 only, iterations40-49, full owners0-5 matched to peers8-13.
Per-rank GPU timestamps were aligned using each trace's origin_ns, subtracting
the common integer origin before float conversion. RS calls were matched by PG
participants, payload metadata, and ordinal within the iteration. New NEP paired
RS completion skew is -0.021 to +0.093ms, small against the observed delays.
This is a consistency check, not independent inter-node clock calibration.

Means across the same60 owner/iteration samples; all milestones below are
relative to that full owner's last expert backward completion:

| Milestone, ms | Healthy | Delayed NEP | Prefetch NEP |
|---|---:|---:|---:|
| Local final expert RS input ready | 0.418 | 65.241 | 71.549 |
| Peer last expert backward complete | 16.711 | 93.955 | 101.133 |
| Peer final expert RS input ready | 17.383 | 97.741 | 104.333 |
| Final expert RS complete locally | 25.170 | 102.204 | 108.693 |
| Optimizer starts | 35.724 | 122.700 | 131.007 |

For prefetch NEP, all local Gather/staging completes at +71.547ms. The peer's
last backward finishes another29.586ms later; its final staging starts1.515ms
after that and takes1.624ms. Peer RS input is ready at +104.333ms. Only4.359ms
remains until local final expert RS completion. This is post-readiness completion
time, NOT pure wire/kernel time. Final-RS peer wait after local RS readiness is
32.784ms (range23.023-40.884ms), versus4.359ms post-readiness (3.951-5.611ms).
Healthy has17.0ms of final-RS peer wait and7.8ms post-readiness completion;
its last bucket has a different payload and this is not a bandwidth comparison.

The additional95.283ms owner post-backward tail versus healthy partitions along
these milestones as follows, without adding overlapping local Gather intervals:

| Timeline component | Additional time, ms |
|---|---:|
| Peer last expert backward finishes later relative to owner | +84.422 |
| Peer last backward to final expert RS readiness | +2.528 |
| Both final expert RS inputs ready to completion | -3.427 |
| Final expert RS completion to final dense RS completion | +5.809 |
| Final dense RS completion to optimizer start | +5.950 |
| Total | +95.283 |

This is a temporal decomposition, not proof that peer GEMMs alone caused84ms
of extra runtime. The first row includes accumulated skew and dependencies.
The reduced replica already starts final backward77.673ms later, versus14.205ms
in healthy (old NEP64.468ms). It then finishes101.133ms later. This is why late
reduced-replica readiness is the dominant observed final-sync limiter.

The final dense RS also waits for reduced ranks: latest input is rank12 in9/10
iterations, rank11 in1/10. On full owners, its input-ready-to-latest-peer-ready
interval averages103.660ms; completion after all inputs are ready is3.169ms.
Final dense RS completes10.417ms after final expert RS, followed by11.897ms
before optimizer.step. These are separate from expert RS transfer.

Do not label the185.912ms sum of owner Gather ranges as transfer time. For
example, owner0/iteration44 Gather bucket2 has both participant inputs ready
at -117.833/-118.510ms relative to owner last backward, but ends at -68.134ms,
just0.366ms after older RS bucket0 completes at -68.500ms. Gather's host call
was submitted at -122.028 to -121.905ms, so it was not submitted late by CPU.
The same example's RS0 waits80.799ms for reduced peer8 input, then8.024ms
to complete. This supports queued/serialized Gather progress, not50ms of
necessary NVLink payload transfer. Exact GPU dependency/resource contention
cannot be distinguished from these event ranges alone.

All expert RS buckets summed after both inputs are ready: healthy52.938ms,
old NEP62.045ms, prefetch61.572ms. Bucket counts/layouts differ (7 versus8),
and these sums can include queueing; do not add them to tail milestones or
interpret them as pure transfer/exposed durations.

Conclusion: (1) dominates the directly resolved final synchronization tail;
(3) is much smaller. (2) includes substantial queueing and is not an independent
185.9ms wire cost. The additional75.6ms final-backward expert envelope is not
fully causally partitioned by this targeted profiler; native model-EP, dense
backward kernels, and CPU gaps would need finer attribution to settle that.
