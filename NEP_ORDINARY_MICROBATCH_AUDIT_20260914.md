# Why Some Ordinary Microbatches Are Slower on Reduced EP

## Scope and Variance

Existing full PyTorch profile steps 10-13, same jobs as the microbatch phase
figure; no new jobs or runtime changes. Each iteration contributes its mean
later-forward or earlier-backward time. The figure's whiskers are sample SD,
not confidence intervals. Four profiled iterations are not enough to infer a
population effect or remove differential profiler overhead.

Representative-rank elapsed time, mean +/- sample SD, ms:

| Case | Healthy rank0 | NEP full rank0 | NEP reduced rank8 |
|---|---:|---:|---:|
| a500 MBS1 later forward | 63.99 +/- 1.72 | 66.07 +/- 0.88 | 68.29 +/- 0.75 |
| a500 MBS1 earlier backward | 55.40 +/- 2.92 | 50.82 +/- 1.85 | 65.75 +/- 3.26 |
| a770 MBS1 earlier backward | 81.07 +/- 4.05 | 77.38 +/- 4.33 | 86.89 +/- 1.64 |
| a770 MBS2 later forward | 115.87 +/- 1.45 | 124.33 +/- 0.64 | 128.23 +/- 1.40 |

The reduced-minus-full difference is positive in every sampled iteration for
these four cases. Independent healthy-run comparison is less decisive for
a770 MBS1 backward because its healthy samples are variable. Neighboring ranks
confirm the same direction and almost the same mean spans:

| Case | Healthy rank1 | NEP full rank1 | NEP reduced rank9 |
|---|---:|---:|---:|
| a500 MBS1 later forward | 64.01 | 65.99 | 68.19 |
| a500 MBS1 earlier backward | 55.54 | 50.95 | 65.92 |
| a770 MBS1 earlier backward | 80.22 | 77.79 | 86.77 |
| a770 MBS2 later forward | 115.77 | 124.34 | 127.98 |

By contrast, a770 MBS2 earlier backward is 118.82 +/- 3.48 on reduced versus
118.76 +/- 0.94 on full. a8b earlier backward is 606.53 +/- 42.46 on reduced,
602.51 +/- 34.09 on full and 606.24 +/- 34.46 on healthy. Neither supports a
stable reduced-replica slowdown from these samples. a500 MBS3 reduced is not
slower than both full and healthy in either ordinary phase.

## Additive GPU-State Accounting

Within the previously defined elapsed windows, GPU activity was partitioned
into exclusive states, including idle (no kernel, memcpy or memset active).
States sum back to each original phase span within 0.002 ms. These are observed
timeline categories, not counterfactual causal costs. Kernel-duration sums and
inclusive CPU scopes are kept separately and must not be added to these states.

Reduced rank8 minus full rank0, ms per ordinary microbatch:

| Case | Total delta | Expert GEMM busy | Other compute busy | Model-EP polling | Model-EP data kernels | GPU idle |
|---|---:|---:|---:|---:|---:|---:|
| a500 MBS1 forward | +2.21 | +0.33 | +1.04 | +1.09 | -0.76 | +0.52 |
| a500 MBS1 backward | +14.93 | +0.71 | +0.17 | +14.13 | -1.09 | +1.02 |
| a770 MBS1 backward | +9.51 | +1.00 | -0.19 | +10.46 | -1.06 | -0.71 |
| a770 MBS2 forward | +3.90 | +0.71 | -0.11 | -3.48 | -1.75 | +8.53 |

Small memory/overlap states and rounding account for less than 0.02 ms per row.
Model-EP data-kernel duration is not a bandwidth measurement.

## Backward: Late Peer Submission, Not Primarily GEMM Service Time

Matched HybridEP device_sync_kernel events by microbatch and ordinal across
all six reduced-replica ranks, verifying matching counts and trace time origins.

- a500: rank8 polls for 19.446 ms per earlier backward microbatch. 19.319 ms
  precedes the latest peer's polling-kernel start; only 0.127 ms follows it.
  On the recorded cross-rank clocks, 17.985 ms precedes the corresponding latest
  peer's CUDA launch-call entry. All 384 six-rank matched intervals intersect.
- a770: rank8 polls for 24.317 ms per earlier backward microbatch. The observed
  time preceding the latest peer's launch call is 23.022 ms. Do not claim an
  exact six-rank post-arrival decomposition: 227/480 matched intervals fail
  strict intersection, by at most 4.83 microseconds. This small discrepancy
  could reflect clock alignment or signaling preceding polling-kernel entry.
  It limits microsecond precision, not the observed multi-millisecond late
  submissions. No clock-offset correction was invented or applied.

Concrete examples, relative to rank8's polling-kernel start:

| Example | Latest peer | Peer grouped-backward CPU ends | Peer launch call begins | Peer poll starts |
|---|---:|---:|---:|---:|
| a500 step13, MB1, polling ordinal26 (zero-based) | rank9 | +3.502 ms | +4.461 ms | +4.484 ms |
| a770 step10, MB1, polling ordinal46 (zero-based) | rank9 | +3.999 ms | +4.938 ms | +4.980 ms |

The associated _GroupedLinearBackward CPU scopes take 2.420 and 3.164 ms.
CUDA runtime/driver interval unions inside those exact scopes are only 0.305
and 0.381 ms. The remaining 2.115 and 2.783 ms is outside recorded CUDA APIs.
Their aten::split_with_sizes children take 1.052 and 0.721 ms. There is no long
CUDA synchronization API explaining either selected grouped-backward scope.

Thus the trace supports ordinary expert-path host preparation/submission delays
propagating into model-EP peer waits. It does not establish whether time outside
CUDA APIs is useful host work, profiler overhead, OS scheduling or some mixture.
The last peer varies; it is not a permanent rank9 bottleneck.

More experts do produce more GEMM launches, but GEMM time is not the dominant
observed backward-span difference. a500 launches 1,056 versus 768 expert GEMM
kernels per backward; their summed duration is 8.561 versus 7.598 ms. a770
launches 1,320 versus 960; summed duration is 12.483 versus 10.769 ms. These
small increments cannot explain the 14.93/9.51 ms reduced-minus-full gaps alone.

## a770 MBS2 Forward: Ordinary Grouped-Expert Host Work Is Exposed

Compared with full rank0, reduced rank8 has 8.530 ms more GPU-idle time but
3.479 ms less model-EP polling and 1.747 ms less model-EP data-kernel time.
Those savings offset much of the idle increase, leaving 3.901 ms more elapsed
forward time. Versus healthy, idle rises 14.438 ms and the span rises 12.359 ms.

- Summed _GroupedLinear CPU scope time rises 21.997 -> 28.878 ms.
- GPU idle overlapping those scopes rises 16.301 -> 22.473 ms, accounting for
  6.172 of the 8.530 ms extra idle versus full. This is a subset, not another
  cost to add.
- Idle before the next GPU activity's host launch call rises 26.301 ->
  32.214 ms. This overlaps the previous measurement and is not additive.
- Expert GEMM launches rise 480 -> 660 (16 -> 22 real local experts across
  30 expert linear calls), while summed GEMM time rises only 5.713 -> 6.405 ms.
- Example: rank8 step10 MB2 has a 0.978 ms idle gap starting 310.442 ms into
  the step. The next nvjet GEMM's launch call begins 0.957 ms into the gap,
  inside _GroupedLinear. The GPU had no work executing and that next GEMM had
  not yet been submitted for almost the entire gap.

This is ordinary per-microbatch expert-path overhead, not a repeat of post-update
NEP parameter redistribution. Increased local expert count gives a concrete
reason for more per-expert work and launches, but this experiment does not
isolate it from profiling/host effects or prove an intrinsic GEMM-efficiency
explanation for the full gap.

## a500 MBS1 Forward: A Small Mixed Difference

The 2.21 ms reduced-minus-full gap is distributed across other compute
(+1.04 ms), model-EP polling (+1.09 ms), expert GEMMs (+0.33 ms), and idle
(+0.52 ms), partly offset by shorter model-EP data kernels (-0.76 ms).
There is no single dominant source, and no basis to label this whole gap a
GEMM-efficiency penalty. The aggregate other-compute increment has not been
attributed to individual kernel families in this bounded audit.

## Artifacts and Limits

Lustre: analysis/nep_microbatch_phases_20260914/ordinary_audit contains 18
rank-level ordinary-phase audits and all-six-rank poll matches for two jobs.
Reproduction scripts: analyze_nep_ordinary_microbatches.py (also --extra-ranks),
and analyze_nep_ordinary_poll_arrivals.py under scripts/nonuniform.

These results explain the observed trace-level mechanisms. They do not prove
the same magnitudes without profiling or establish a population significance
test from four temporally adjacent iterations. No additional benchmarks ran.
