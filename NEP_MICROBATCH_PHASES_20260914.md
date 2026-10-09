# First Forward and Final Backward Microbatch Comparison

Companion to `small_model_parity_final_ga`, using the same five configurations:
a500 MBS1/3 (2994105/2994279), a770 MBS1/2 (2995392/2995524), and a8b MBS1
(2994648). No jobs launched and no runtime changes.

## Sampling

- Existing full PyTorch profiles, iterations 10-13 only. No post-profile timing
  is mixed into the bars. Profiling can distort healthy/NEP differently.
- Representative healthy rank 0, NEP full-owner rank 0, and NEP reduced-owner
  rank 8 (small models) or 16 (a8b). These are not all-rank averages.
- Each bar averages four iterations; error bars show sample standard deviation
  across iterations. The rest category first averages non-special microbatches
  within each iteration, so its unit remains time per microbatch.
- No parameter-CPU-cleanup branch results are included.

## Timing Definitions

Backward microbatches are identified from autograd evaluate-function scopes,
split at inter-microbatch gaps. Every microbatch must contain exactly two
grouped-linear backward calls per MoE layer. Boundary gaps must exceed every
internal gap. GPU operations are assigned by CPU external-ID correlation,
with CUDA launch-correlation fallback.

There are no explicit model-forward scopes in these traces. Forward plus
preparation means GPU activity submitted between the preceding backward CPU
completion (step start for the first microbatch) and the next backward CPU
start. Its elapsed span is first associated GPU activity start to last associated
GPU activity completion. This includes parameter synchronization and preparation;
it excludes host delay before the first associated GPU activity starts.

Backward execution spans correlated autograd GPU work, including native token
communication, gradient accumulation, and internal delays. Identified gradient
collectives and NEP staging operations do not define its endpoints. Their
interference can still increase the span; excluding their kernels does not
remove their causal effects.

Finalization tail is final backward GPU completion to the first optimizer GPU
activity. It includes unfinished synchronization, checks and host launch gaps,
not just NEP communication. It is shown separately so exposed synchronization
is not described as extra backward computation.

These elapsed envelopes are not kernel-duration sums or a mutually exclusive
decomposition of iteration time. GPU-idle and non-NCCL compute-sum measurements
are retained in the JSON for further inspection; the figure plots elapsed time.

## Validation

All 15 rank traces / 60 rank-iterations passed:

- Four expected profile steps, correct GA and grouped-backward counts.
- Separated backward clusters; all GPU activities correlated.
- Nonempty forward/backward GPU assignments and a nonnegative finalization tail.
- No parameter-redistribution calls in later forward windows.
- Final-GA non-NCCL compute sums agree with the existing final-GA analysis
  within 0.002 ms for every sampled rank-iteration.

The observations support a first-forward-specific cost, but are not an isolated
measurement of CPU overhead or a direct experiment increasing GA. Later forward
and earlier backward spans are not guaranteed identical between healthy and NEP.

## Artifacts

Data, per-microbatch samples, PNG and PDF:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_microbatch_phases_20260914/`.

Reproduce with:

```bash
PYTHONPATH=/tmp/nep-analysis-deps python3 scripts/nonuniform/analyze_nep_microbatch_phases.py
python3 scripts/nonuniform/plot_nep_microbatch_phases.py
```

The optional temporary dependency path provides ijson for efficient streaming;
the analyzer falls back to the existing repository trace parser without it.
