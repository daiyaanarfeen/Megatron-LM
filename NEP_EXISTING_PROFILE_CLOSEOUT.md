# NEP existing-profile results

Completed 2026-09-09 for the scope accepted by the user: explain the observed
before/after gaps using existing measurements, disclose confounds, and do not
claim complete causal attribution or reproducibility of an unstable mean.
No additional training was launched for this closeout. Pending jobs 2979170 and
2981040 were canceled before allocation. Training implementation is unchanged.

## Figures

Published set: [Talos figures](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/).

1. [a8b before/after](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/a8b_overview.png): original timing, exact phase stacks, signed gap decomposition, and fixed layout work.
2. [a8b communication](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/a8b_communications.png): exposed and overlapping category activity.
3. [a500 before/after](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/a500_overview.png): original timing, exact phase stacks, and full/reduced rank progress.
4. [a500 communication](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/a500_communications.png): exposed and overlapping category activity.
5. [a500 reproducibility](https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_existing_evidence_20260909/a500_reproducibility.png): every original/repeat timing sample, preserving the mean-parity discrepancy.

## What the comparisons show

| Workload setting | Logged H / NEP ms | Mean parity | Profiled H / NEP ms | Profile gap ms |
| --- | ---: | ---: | ---: | ---: |
| a8b GA4 | 4010.13 / 4778.03 | 83.93% | 4534.87 / 5636.59 | 1101.72 |
| a8b GA20 | 20306.30 / 21125.60 | 96.12% | 23343.63 / 24424.72 | 1081.09 |
| a500 GA12/12 | 1205.73 / 1471.73 | 81.93% | 1751.73 / 2263.37 | 511.63 |
| a500 GA12/9 | 1206.70 / 1238.30 | 97.45% | 1712.66 / 1829.73 | 117.07 |

Logged means use iterations 8--10. Profile breakdowns use steps 5--6. These are
different windows: the profile milliseconds do NOT decompose the logged gap.
Parity means healthy owner iteration time divided by full-replica NEP owner
iteration time, not aggregate token-throughput parity. Profile tables use rank0;
they are not sums over ranks or worst-rank collective residency.

### a8b: more work amortizes a persistent absolute penalty

The logged gap is 767.90 -> 819.30 ms while healthy time grows about fivefold.
The profiled gap is 1101.72 -> 1081.09 ms. Thus the improvement is predominantly a
smaller relative penalty, not disappearance of NEP operations.

Measured fixed-count work supports this interpretation:

- Both GAs issue eight NEP gradient Gathers and expert gradient reduce-scatters
  per iteration, not per microbatch.
- Parameter redistribution contains 4500 device copies per iteration at both
  GAs; its recorded GPU scope spans 197.34 -> 181.88 ms.
- Gradient layout staging contains 2580 copies per iteration at both GAs; its
  scope spans 120.59 -> 105.40 ms. These scopes include gaps and are already inside
  the iteration, not extra costs to add to the total.
- Within those two scopes, combined GPU-idle time before the next correlated
  host launch begins is 152.56 -> 136.02 ms. That particular operation cannot have
  executed earlier because its host launch had not begun. This does not assign
  every idle interval to Python or establish a counterfactual speedup.

The composition of exposure also changes. On owner 0, NEP expert gradient RS
exposure falls 331.02 -> 27.65 ms, but model-EP polling exposure grows 368.06
-> 4255.04 ms over an iteration with five times as many microbatches. Healthy
polling exposure also grows 142.89 -> 3738.38 ms; the relevant excess is 225.18
-> 516.66 ms, NOT the whole NEP polling duration. Reducing one exposed category
does not imply the entire iteration improves by that duration.

**Placement limitation:** GA4 job 2863174 used one NVLink domain (nvlblk22);
GA20 job 2877884 used two (nvlblk26/27). Replica-local EP was within its domain
in both, but inter-replica EDP links changed. Both requested segment4. Therefore
this is observed amortization, not an isolated causal experiment on GA.

### a500: less reduced-replica work removes the owner's long gradient drain

Both original settings ran on the same four GB300 nodes, theia0028--0031, in one
NVLink domain. Full-replica MBS1/GA12 is unchanged; reduced GA changes12 ->9.

| Owner0 phase delta, NEP minus healthy | GA12/12 ms | GA12/9 ms |
| --- | ---: | ---: |
| Iteration start through model backward completion | 330.70 | 382.41 |
| Model backward end to GPU Adam start | 171.97 | -272.18 |
| GPU Adam span | 0.31 | 1.49 |
| After GPU Adam | 8.65 | 5.35 |
| **Sum: complete profiled iteration gap** | **511.63** | **117.07** |

The 394.56 ms gap reduction is located mainly in a 444.16 ms improvement in the
gradient-drain delta, offset by 51.71 ms MORE through-backward delay and small
optimizer/tail changes. Owner backward did not become faster.

NEP owner 0's absolute gradient drain falls 514.74 -> 12.27 ms. The reduced peer's
drain increases from about 12.7 to 188.1 ms: waiting moves to the earlier reduced
replica. Owner expert gradient RS residency falls 604.88 -> 1.69 ms while the
reduced peer's increases 1.68 -> 281.96 ms. This is consistent with the intended
work-balancing effect, not a claim that communication disappeared globally.

Why proportional work can still be slower has supporting, but incomplete,
evidence. EP8 owns 16 experts/GPU; EP6 owns 21 or 22. A diagnostic profile counted
9216 backward GEMM launches on a healthy 16-expert rank versus 12096 on a
21-expert reduced rank, exactly 21/16 times as many. On two reduced ranks with
the SAME 21 experts and 12096 launches, launch-API residency differed 60.18 versus
113.93 ms; grouped backward CPU time also differed. Therefore launch count is a
real additional burden but is not a complete explanation of rank variability.
Those CPU-binding diagnostic measurements are supporting evidence, not the
original before/after timing window. CPU binding was not adopted as a fix.

### a500 reproduction limits the high-parity claim

| Reduced GA | Original mean / median parity | Exact-node repeat mean / median parity |
| --- | ---: | ---: |
| 12 | 81.93% /82.04% | 89.36% /88.83% |
| 9 | 97.45% /94.41% | 94.30% /94.25% |

The original below-proportional healthy samples were 1158.5, 1169.7, 1291.9 ms.
The slower 1291.9 ms sample inflated mean parity. Below-proportional median parity
reproduced closely, but the original 97.45% mean did not. We retain the original
data, do not cherry-pick a replacement, and do not assert stable >=95% parity.

## Exact activity accounting, without treating overlap as a cause

Each row below is a DISJOINT rank0 GPU activity state. Single-category rows mean
that category alone is active; concurrent intervals are counted once in their
own row. Numbers are NEP minus healthy, averaged over profile steps5--6.

| Exclusive state delta (ms) | a8b GA4 | a8b GA20 | a500 GA12/12 | a500 GA12/9 |
| --- | ---: | ---: | ---: | ---: |
| No recorded GPU activity | 456.34 | 351.98 | 13.47 | 65.61 |
| Model-EP polling only | 225.18 | 516.66 | 158.98 | 280.57 |
| Expert gradient RS only | 253.71 | -24.11 | 43.08 | -26.90 |
| Gradient Gather only | 12.22 | 12.15 | 0.00 | 0.28 |
| Parameter redistribution only | 26.30 | 27.04 | 1.48 | 1.18 |
| Other kernels only | 48.55 | 57.34 | 24.39 | 68.96 |
| Concurrent categories, counted once | 8.40 | 100.39 | 228.41 | -285.10 |
| Remaining exclusive states | 71.01 | 39.65 | 41.82 | 12.48 |
| **Complete iteration gap** | **1101.72** | **1081.09** | **511.63** | **117.07** |

Rounding can change displayed sums by0.01ms. Remaining states are individually
identified memory, gradient accumulation/staging, native communication and
parameter gather states; they are not an unmeasured time residual. `Other
kernels` is deliberately NOT named pure compute: unclassified communication
metadata kernels can occur there.

In a500, healthy already has concurrent waiting collectives. Removing the NEP
owner's long wait makes its concurrent-state time much smaller than healthy's,
producing a negative delta. This is NOT evidence that losing useful overlap
improved performance. Conversely, more simultaneous polling is not necessarily
more work hidden behind useful computation. The phase/progress panels establish
where this change occurs; the state table describes what is resident there.

## Interpretation rules and remaining limits

- The communication plots show category interval unions, not sums of overlapping
  same-category kernels. Solid portions have no other classified GPU activity
  on that rank. Hatched portions intersect another category. Hatched portions
  from different rows must not be added together.
- A collective's duration includes participant waiting. It is not a measurement
  of payload-only transfer time or saturated link bandwidth.
- These DistOpt traces perform expert gradient reduce-scatter and parameter
  all-gather. Owner-transfer broadcasts are parameter redistribution, not the
  exposed gradient Scatter used in earlier non-DistOpt experiments.
- Dense-DP sync includes the small reductions actually present in these runs.
  We did not retroactively relabel them as gradient synchronization or remove
  them by changing flags.
- Progress comparisons use each rank's own step origin. No unsynchronized
  cross-host timestamp difference is presented as a precise arrival wait.
- Profiler signatures/phase closure are accounting checks, not proof of
  numerical gradient correctness. This closeout does not modify the previously
  validated training implementation.
- The underlying cause of every participant delay, busy-spin, or runtime
  launch-cost variation remains unresolved. The accepted result explains the
  observed timing composition and improvement, not unique counterfactual costs.

## Reproduce the analysis

Renderer: `scripts/nonuniform/render_nep_existing_evidence.py`.
Processed all-rank traces remain in
`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_owner_progress_20260909`.
Figures and machine-readable inputs/provenance (`evidence.json`) are under
`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_existing_evidence_20260909`.
Only PNGs are published to the Talos directory.

```bash
PYTHONPATH=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/python_plot_tools \
MPLCONFIGDIR=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_owner_progress_20260909/mplconfig \
python3 scripts/nonuniform/render_nep_existing_evidence.py \
  --root /lustre/fsw/coreai_comparch_sysarch/darfeen \
  --output /lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_existing_evidence_20260909
```

The renderer rejects unexpected profile windows, checks exact phase/state
closure and exposure bounds, and checks that signed phase deltas equal the
profiled end-to-end gap. Existing CPU-only accounting tests also pass.
