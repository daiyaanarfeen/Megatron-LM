# NEP owner-progress analysis (2026-09-09)

**Final existing-data scope accepted 2026-09-09.** See
[the completed results and figures](NEP_EXISTING_PROFILE_CLOSEOUT.md). This file
is retained as the detailed historical investigation. Its earlier completion
verdicts are superseded by that report. Exact-node a5002979126 completed; pending
a8b2979170 and routing-control2981040 were canceled when further jobs were no
longer needed for the accepted scope. Placement confounds remain explicitly
disclosed rather than claimed resolved by a run that never occurred.

## Placement audit update

| Original case | GB300 nodes | NVLink blocks | Consequence |
| --- | --- | --- | --- |
| a500 GA12/12 and12/9,2977274 | theia0028--0031 | nvlblk18 | Same allocation and domain across both settings |
| a8b GA4,2863174 | theia0091--0092,0094,0100--0102,0106--0107 | nvlblk22 | Both replicas within one domain |
| a8b GA20,2877884 | theia0176--0178,0180,0187,0189,0196--0197 | nvlblk26 and27 | EDP crosses domains; replica-local EP remains within a domain |

Both original a8b jobs requested segment4, but this does not force distinct
segments into distinct NVLink domains. Their EDP placements are not equivalent.
The replacement runs both GA settings on the exact GA20 allocation.

The new CPU-placement diagnostic2979045 was canceled: Slurm's automatic core
binding put a partially occupied node's GPU1 process on the wrong NUMA node.
Its completed unbound pair is GB200, not a reproduction of the GB300 result.
Actual unbound NEP rank placement is4/4/3/3 across four nodes, not4/4/4/2.
Do not assume ranks8 and11 share a host. A provisional journal claim based on
that assumption has been withdrawn; no cross-host timing bound is accepted
without verified clock alignment. Existing per-rank durations remain usable.

This follow-up replaces the ambiguous CPU-phase and generic EDP/Scatter labels in
the earlier failure-category plots. No training implementation was changed.
The longer confirmation is reported separately from the historical traces.

**Closeout verdict:** retain a8b as an observed amortization example; withdraw
a500 as a demonstrated general failure/fix category. The historical a500 trace
does show a readiness-wait shift, but its strong clean-timing improvement was not
reproduced consistently. This closes the bounded evidence review, not a claim
that all NEP critical-path costs have been causally identified. See the final
closeout section for the additional all-rank and host-submission evidence.

## Measurement contract

- Logged timing and profiled timing are separate measurements. Do not use profile
  milliseconds as an exact decomposition of the unprofiled logged timing gap.
- Parity is healthy owner iteration time / NEP full-replica owner iteration time.
  It is not equal aggregate throughput: healthy and NEP global batch sizes differ.
- Model-backward completion is the last correlated non-communication model-backward GPU operation,
  excluding AccumulateGrad hooks and NEP staging. GPU Adam annotations identify
  actual optimizer execution; CPU Adam annotations identify submission only.
- Four disjoint phases close exactly to the profile-step span: start through
  model-backward completion, gradient drain to GPU Adam, GPU Adam, and after Adam.
  These locate delay; they are not four independent root causes.
- Collective residency includes participant waits. Gradient reduce-scatter,
  parameter all-gather, gradient Gather, and parameter redistribution are separate.
  In these DistOpt traces the owner-transfer broadcasts redistribute parameters;
  they are not end-of-backward gradient Scatters.
- Native dense-DP synchronization is labeled broadly: a8b also has one integer
  and three two-float all-reduces, and a500 has one integer all-reduce, in addition
  to three dense gradient RS calls. Their operation names, payloads, and times are
  retained individually; they must not all be called gradient RS. Existing
  benchmark flags were preserved rather than silently disabling these calls.
- Operation overlap uses interval unions and exclusive category masks. Polling
  alongside another collective is not automatically useful compute overlap.
  Category activity uses the union of its GPU intervals, not a sum that counts
  simultaneous same-category kernels twice; per-call durations remain available.
- Rank progress uses each rank's own step origin. No unsynchronized cross-rank
  timestamp subtraction or minimum-participant "true transfer time" is used.

## Category 1: a8b amortization

Model: a8b_120b_latentmoe_1t, model H4608, expert input/latent width 1152,
expert F3072, 512 experts, top-k 6,
sequence 8192, TP2/CP1/ETP1, physical EP16/EP12, MBS2 on both replicas.
Healthy has six observed expert gradient RS calls and NEP eight per iteration.
GA4 and GA20 are separate GB300 allocations; this is not an isolated overlap
intervention or an iso-bucket experiment.

| GA on both replicas | H / NEP GBS | Logged H / NEP ms | Logged parity | Profile gap ms |
| --- | --- | --- | --- | --- |
| 4 | 128 / 112 | 4010.13 / 4778.03 | 83.93% | 1101.72 |
| 20 | 640 / 560 | 20306.30 / 21125.60 | 96.12% | 1081.09 |

The logged absolute gap is 767.90 -> 819.30 ms while healthy time grows about
fivefold. This supports amortization, not disappearance of NEP work.

Rank-0 profile gap by disjoint location:

| Location | GA4 delta ms | GA20 delta ms |
| --- | ---: | ---: |
| Through first microbatch backward completion | 495.37 | 465.87 |
| Middle microbatches, excluding last | 79.50 | 384.46 |
| Last microbatch | 501.62 | 179.06 |
| After model-backward completion | 25.23 | 51.70 |

The last-microbatch gap shrinks, but recurring middle-microbatch delay grows.
It is incorrect to describe the result simply as all extra EDP becoming hidden.

### Direct evidence for fixed layout/submission work

Rank-0 mean GPU scope measurements; these are contained within the phase table,
not additional costs to add to it:

| NEP scope | Copies/iter | GA4 / GA20 span ms | Copy activity ms | Broadcast activity ms | No recorded GPU activity ms |
| --- | ---: | --- | --- | --- | --- |
| Parameter redistribution | 4500 | 197.34 / 181.88 | 18.21 / 18.14 | 26.30 / 27.04 | 152.82 / 136.70 |
| Gradient staging | 2580 | 120.59 / 105.40 | 16.02 / 16.04 | 0 / 0 | 104.57 / 89.36 |

Of the idle intervals, 92.04 / 84.06 ms in redistribution and 60.52 / 51.96 ms
in staging occur before the next GPU operation's correlated host launch begins.
That operation could not execute earlier because it had not been submitted.
The unchanged implementation issues per-expert/per-slot copy_ calls in layout
packing/unpacking loops, consistent with this measured launch burden.
The copy counts also match the layout: 28 main-stack MoE layers plus two MTP
MoE layers, two weights per expert, and 43 logical owner experts on rank 0 give
30 * 2 * 43 = 2580 gradient-staging copies. Parameter redistribution packs those
2580 owner slots and unpacks 30 * 2 * 32 = 1920 physical slots, totaling 4500.
Runtime arguments confirm latent width 1152 and two MTP layers; do not interpret
model H4608 as the expert GEMM input dimension.
The remaining idle cannot all be assigned to Python from this evidence alone.
Profiling also adds host overhead, so these are not exact clean-run penalties.

This explains a material fixed component, not every millisecond: upstream causes
of all model-EP polling and the difference in last-microbatch gradient RS waiting
are still not uniquely identified by these traces.

Both a8b and a500 runtime logs enable moe_router_force_load_balancing but disable
moe_router_force_uniform_routing. The former replaces logits with random logits;
it is not an invariant expert/peer split map. This setting was preserved for the
confirmation. It is a possible source of variance, not a proven cause of the
observed timing drift. Profiler shape and call-stack collection are disabled.

### HybridEP dynamic-sizing synchronization is still present

The runtime setting moe_expert_rank_capacity_factor=None selects dynamic receive
sizing. The trusted token_dispatcher.py explicitly documents CPU synchronization
in this mode; it is not the available static-budget path. Actual traces contain
cudaStreamSynchronize calls with HybridEPDispatch as their CPU parent.

Inside rank 0's final-microbatch GPU-progress window:

| Measurement (ms) | GA4 healthy | GA4 NEP | GA20 healthy | GA20 NEP |
| --- | ---: | ---: | ---: | ---: |
| Microbatch duration | 1165.82 | 1667.44 | 1173.24 | 1352.31 |
| Host blocked in HybridEPDispatch sync (union) | 575.11 | 839.74 | 614.30 | 644.08 |
| Expert gradient RS concurrent with that host wait | 75.13 | 329.68 | 64.89 | 112.72 |
| Expert gradient RS with no other recorded GPU work | 77.30 | 331.02 | 51.76 | 27.65 |

These waits prevent the same host thread from submitting later work until they
return. They are not additive to GPU time, and temporal intersection does not
prove an explicit EDP-to-model stream dependency or a 330 ms counterfactual saving.
The code preserves async native DDP and disables its gradient validation checks.
Thus "async EDP" does not mean unlimited future-compute submission in this dynamic
dispatcher workload. The cause of each long synchronization still depends on its
GPU predecessors and participant progress; those are not all uniquely resolved.

Both GA cases issue eight NEP Gather and expert gradient RS calls per iteration,
not eight per accumulated microbatch. More GA amortizes these fixed-count actions;
it does not by itself create a proportionally longer final-backward overlap window.
No capacity/routing/stream settings were changed during this task.

## Category 2: historical a500 readiness result

Model: scaling_ladder/a500m_2b_moe_140b, H1024, expert F640, 128 experts,
top-k 6, sequence 8192, TP1/CP1, EP8/EP6, MBS1. Healthy GA12/12;
NEP GA12/12 or GA12/9. Healthy has four expert gradient RS calls, NEP eight.
Both historical cases are from GB300 job 2977274 on the same allocation.

| NEP GA | H / NEP GBS | Logged H / NEP ms | Mean / median parity | Profile gap ms |
| --- | --- | --- | --- | --- |
| 12/12 | 192 / 168 | 1205.73 / 1471.73 | 81.93% / 82.04% | 511.63 |
| 12/9 | 192 / 150 | 1206.70 / 1238.30 | 97.45% / 94.41% | 117.07 |

The old three-sample mean does not establish reliable >=95% parity.

### The apparent post-optimizer delay was GPU gradient drain

Rank-0 mean profile phases:

| Case | Through backward ms | Gradient drain ms | GPU Adam ms | After GPU Adam ms |
| --- | ---: | ---: | ---: | ---: |
| Healthy reference for 12/12 | 1388.08 | 342.77 | 0.96 | 19.92 |
| NEP 12/12 | 1718.78 | 514.74 | 1.27 | 28.57 |
| Healthy reference for 12/9 | 1407.86 | 284.46 | 0.96 | 19.38 |
| NEP 12/9 | 1790.27 | 12.27 | 2.46 | 24.73 |

CPU Adam submission finishes hundreds of milliseconds before GPU Adam starts in
the slow cases. A subsequent scalar read synchronizes the compute stream while
gradient RS is still outstanding; it does not represent hundreds of milliseconds
of optimizer computation or a mysterious post-optimizer operation.

With NEP 12/12, rank 0 completes backward at 1718.78 ms and drains for 514.74 ms;
its reduced peer rank 8 completes at 2220.8 ms and drains for only 12.7 ms.
With 12/9, rank 0 drains for 12.27 ms while rank 8 drains for about 188.1 ms.
The direction of readiness asymmetry reverses. Grad RS residency correspondingly
changes from 604.88 -> 1.69 ms on rank 0 and 1.68 -> 281.96 ms on rank 8.
Waiting moved to the early reduced peer; it did not vanish across the job.

The profile gap falls by about 394.6 ms, explained *temporally* by a 444.2 ms
improvement in the gradient-drain delta, offset by 51.7 ms more through-backward
delta and small Adam/after-Adam changes. Backward did not get faster on rank 0.
Healthy already contains substantial replica skew; its own gradient drain offsets
some NEP delay. Exact upstream causes of every model-EP poll remain unresolved.

## Reproduction and artifacts

Analysis scripts: scripts/nonuniform/analyze_nep_owner_progress.py,
analyze_nep_scope_progress.py, render_nep_owner_progress.py, and
test_nep_owner_progress.py. Historical trace manifests remain linked from the
earlier audit. New derived data and plots are under:

`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_owner_progress_20260909/`

The confirmation uses job 2977799, four GB200 nodes, segment=4, two rounds with
reversed workload and healthy/NEP order. Each case runs 35 iterations; logged
timing uses 11--30 and all-rank profiling captures 32--33. This is a new hardware
allocation with longer warmup than the historical GB300 run, not an isolated
test of warmup versus node placement. Completed results follow below.

## Longer confirmation results

The first allocation (2977799) was preempted after 30:05 during its last healthy
launch. Its three complete pairs remain valid; the incomplete pair is excluded.
Recovery job 2978113 ran the two reverse-order pairs in 16:30 with exit code 0.
Both allocations used four GB200 nodes and segment=4. Recovery nodes were
lyris[0080,0083,0085-0086], versus lyris[0038-0039,0042,0046] in the first job.
Each comparison shares nodes, but the complete recovery dataset is not a single
allocation eight-case experiment. Training implementation hashes match before
and after recovery.

Each table cell is mean +/- sample standard deviation from 20 iterations (11--30).
Every completed case has all-rank traces from iterations 32--33.

| Job / round | NEP GA | Run order | Healthy ms | NEP ms | Owner parity |
| --- | --- | --- | --- | --- | --- |
| 2977799 / 1 | 12/12 | H then N | 1210.47 +/- 18.32 | 1259.95 +/- 6.72 | 96.07% |
| 2977799 / 1 | 12/9 | H then N | 1224.72 +/- 4.32 | 1286.11 +/- 32.90 | 95.23% |
| 2977799 / 2 | 12/9 | N then H | 1254.67 +/- 3.16 | 1236.75 +/- 3.58 | 101.45% |
| 2978113 / 2 | 12/9 | N then H | 1259.39 +/- 2.29 | 1319.53 +/- 6.05 | 95.44% |
| 2978113 / 2 | 12/12 | N then H | 1208.80 +/- 3.41 | 1282.05 +/- 10.35 | 94.29% |

Do not average away the run effect: on the first allocation, healthy proportional
timing rises from about 1198 to 1242 ms within the sample window; reduced GA9 rises
from about 1234 to 1310 ms. Sample SD is descriptive, not proof of stationary noise
or an independent-samples confidence interval.

The historical 81.93 -> 97.45% improvement is **not reproduced as a stable effect**.
Proportional now reaches 94.29--96.07%; GA9 reaches 95.23--101.45%, with overlapping
run-level variation. It is not justified to present GA9 as a generally necessary
fix for this workload from these data.

Profiling changes both absolute time and the observed gap. For example, first-round
proportional has a logged gap of 49.48 ms but about 151.26 ms between the two mean
profile-step spans. First-round GA9 has a logged gap of 61.40 ms but about 280.49 ms
in the profile spans. Those profiles explain the recorded execution, not an exact
clean-run millisecond decomposition. All-rank recording can perturb CPU/runtime scheduling
despite shape/stack collection being off; a specific share attributable to that
instrumentation is not established by this experiment.

## What is and is not explained

- Supported: fixed layout/copy/submission work is a material a8b cost that can be
  amortized by more accumulation. Its nominal cost did not disappear at GA20.
- Supported: the old a500 post-CPU-Adam span was mostly outstanding GPU gradient
  work, and reducing replica work reversed owner/peer gradient readiness.
- Not supported: all extra native collective activity is transfer rather than
  waiting, every model-EP poll has a uniquely identified upstream cause, or the
  old clean-run improvement is reliably reproduced without confounders.
- Still required for a universal causal claim: isolate routing/runtime drift and
  obtain a representative critical path without treating profiled and logged
  timings as interchangeable. No further model or implementation changes were
  made as part of this analysis.

## Validation and published figures

The five CPU-only analysis tests pass. Trace validation passes 264 owner/peer-step
checks: 96 for a8b, 48 for historical a500, and 120 for confirmation. These check
exact phase/activity accounting and matching ordered expert collective signatures;
they are not numerical gradient validation. Python formatting, shell syntax, and
`git diff --check` also pass.

The [12 published figures](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_owner_progress_20260909/)
include latency comparisons, disjoint GPU-progress phase deltas, owner/peer
timelines, microbatch localization, the gradient dependency example, and timing
drift. All remote PNG checksums match the local files. This completes the planned
analysis and confirmation experiments, not a complete causal attribution of every
remaining model-EP delay.

## Final closeout: equal-count work and host submission

No new GPU jobs or training changes were made for this closeout. The new offline
analysis reads all 60 historical a500 rank traces (two cases, healthy16/NEP14),
using the interval from backward completion of microbatch1 to backward completion
of microbatch8: seven successive forward/backward microbatches on each local clock.
It does not subtract timestamps between ranks. All GPU events are included in the
activity partition, including kernels shorter than 100 us. CPU call unions and
GPU-idle subsets are separate views, never additive overhead components.

### What made the historical a500 reduced replica late?

The reduced replica repeatedly took longer per microbatch, rather than developing
all its lateness at the final gradient collective. Equal-count windows show:

| Seven-microbatch window | Healthy reference ms | NEP reduced rank8 ms | NEP reduced rank11 ms |
| --- | ---: | ---: | ---: |
| GA12/12 case | 989.75 | 1173.03 | 1173.10 |
| GA12/9 case | 950.44 | 1173.12 | 1172.68 |

Reducing GA did not improve the reduced replica's per-microbatch service time.
It removed three microbatches and shifted the final waiting burden to that replica.
The healthy reference also changed by ~39 ms for the same seven microbatches.
Thus the original improvement must not be framed as fixing expert efficiency or
making all NEP communication disappear.

The within-replica delay has a measured host-submission component:

| Local seven-microbatch metric | Healthy rank8 | Healthy rank11 | NEP12 rank8 | NEP12 rank11 | NEP9 rank11 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU idle ms | 158.9 | 159.1 | 189.6 | 465.5 | 537.5 |
| Idle before next GPU launch begins ms | 122.9 | 120.7 | 150.4 | 379.1 | 424.4 |
| Above interval inside grouped-linear CPU calls ms | 78.4 | 76.5 | 85.3 | 174.8 | 182.3 |

The final row is a subset of the second, which is a subset of the first. These
numbers cannot be added to one another or to the peer's polling residency.
They establish that much of the idle time precedes submission of the next GPU
operation. It is not a queued GEMM simply taking hundreds of extra milliseconds.
For context, nvjet kernel-family time in this window is 116.41 ms on healthy
rank11 versus 119.57 ms on NEP12 rank11, while nvjet launch count rises from 9303
to 11823. NEP rank8 has 12327 such launches and 124.17 ms. These are nvjet-family
measurements, not an expert-only FLOP-efficiency measurement, and same-name union
durations must not be treated as a cross-family exclusive partition.

The observable chain is: recurring delayed submission on some EP participants,
model-EP polling on other participants, later reduced-replica gradient readiness,
then long expert gradient RS/gradient-drain residency on the full owner. Matched
owner/peer collective signatures support the final link. The idle/poll pattern
supports the earlier links, but does not uniquely identify which peer delayed
each HybridEP signaling event. We do not assert that rank11 is the unique straggler.

The residual causal question is why those host submissions were late: CPU work
inside grouped-linear calls, scheduling/preemption, library locking, or other
runtime effects. The current traces do not resolve that distinction. Dynamic
HybridEP host synchronization, random-logit routing, and profiled-versus-logged
timing differences remain relevant controls. Removing the sync or changing
routing would be a new intervention, not a justified correction to this dataset.

### Final acceptance decisions

- **a8b: accepted as amortization, not as complete overhead attribution.** Logged
  overhead remains 768--819 ms as the denominator increases about fivefold. The
  fixed 4500-copy redistribution and 2580-copy staging scopes are measured costs;
  they are not claimed to explain the entire ~1.1 s profiled gap. Polling/wait
  categories remain temporal observations where their upstream cause is unresolved.
- **a500: withdrawn as a proven general failure/fix category.** Its historical
  profiled gradient-drain improvement is real, and fewer microbatches reduce the
  amount of persistently slow recurring work. The original 82% -> 97% clean-run
  improvement is not established as a reproducible workload-specific mechanism.
- **No universal causal closure claimed.** The earlier proposed >=90% causal
  attribution gate has not been met. Exact phase closure is not a substitute.
  Further work would require an explicitly scoped host-scheduling/backend-signaling
  investigation, not another generic timing sweep. No additional run was submitted.

The two [final evidence figures](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_closeout_20260909/)
replace the earlier plots as the closeout presentation. Every phase stack closes
to its own profiled gap; logged timing is separate. The a500 recurring-state bars
close to their own seven-microbatch windows. Gray/green host bars are nested
subsets of GPU idle, not additive explanations for the owner iteration gap.

Reproduction: scripts/nonuniform/analyze_nep_closeout.py consumes the existing
category2 manifest and category2_v3 cache; --host-ranks 0,8,11,12,13 adds targeted
host-submission measurements. scripts/nonuniform/render_nep_closeout.py renders
the figures from those outputs and existing owner-progress data. All generated
data are under /lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_closeout_20260909/.

Closeout validation: six CPU-only analysis tests pass, all 120 recurring windows
close exactly, and 40 host-idle checks match the independent raw GPU partition
and obey their subset bounds. Black/isort and diff whitespace checks pass. Both
published PNG hashes match their local originals. These validate the measurements,
not the unresolved causal claims.
