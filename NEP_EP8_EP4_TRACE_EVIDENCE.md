# NEP EP8/EP4 Trace Evidence

## Scope

This report uses all-rank PyTorch traces from Lyris GB200 job `2438661`:

- balanced healthy EP8/EP8
- balanced proportional NEP EP8/EP4
- fixed-bias healthy EP8/EP8
- fixed-bias proportional NEP EP8/EP4

All four cases use the same 14-stage `MEMEM*EMEMEM*E` model, 128 experts,
TP2, ETP1, MBS2, seven microbatches per rank, local CUDA graphs, no optimizer
step, and `ProfilerStep#3`. Clean iteration timing is the mean of iterations
6-8. The healthy and NEP cases for each routing mode ran sequentially in the
same allocation.

Regenerate the measurements with:

```bash
python3 scripts/nonuniform/analyze_ep8_ep4_trace_evidence.py
```

## Proportional Allocation

The full replica has four dense-DP lanes and contributes 56 samples per
iteration; the reduced replica has two lanes and contributes 28. This is a
2:1 batch split for an EP8:EP4 GPU split. It gives both replica types the same
mean routed-token volume per expert GPU, but it does not give them the same
expert-GEMM shape.

All values below are means across the eight full-replica ranks or four
reduced-replica ranks in the balanced proportional trace.

| Trace metric | Full EP8 | Reduced EP4 | Reduced minus full |
| --- | ---: | ---: | ---: |
| Local experts per GPU (`128 / EP`) | 16 | 32 | +16 |
| Routed assignments / dispatch / rank | 49,152 | 49,152 | 0 |
| Per-dispatch routed-load CV | 0.407% | 0.397% | -0.010 pp |
| Expert-stream NVJet kernels / rank | 4,032 | 8,064 | +4,032 |
| Mean expert-kernel duration | 40.430 us | 21.171 us | -19.259 us |
| Expert-kernel residency | 163.014 ms | 170.727 ms | +7.713 ms |
| Native model-EP residency | 123.664 ms | 143.498 ms | +19.834 ms |
| First large dense-DP bucket launch | 875.948 ms | 1017.724 ms | +141.777 ms |
| GPU-active union before that bucket | 763.320 ms | 791.664 ms | +28.344 ms |
| No-GPU-event gaps before that bucket | 112.603 ms | 225.956 ms | +113.353 ms |
| `cudaStreamWaitEvent` calls | 6,005 | 7,307 | +1,302 |
| CUDA graph launches | 280 | 280 | 0 |
| Aggregate `cudaGraphLaunch` API time | 24.723 ms | 34.448 ms | +9.725 ms |

The traces therefore establish the following chain:

1. Routing does not give EP4 more token assignments per GPU. Both sides
   receive 49,152 assignments per dispatch, and both have about 0.4% rank CV.
2. EP4 spreads that same payload over twice as many local experts. Its expert
   stream executes exactly twice as many kernels, while the mean kernel is
   only 52.4% as long. Aggregate expert-kernel residency rises only 4.7%.
3. The reduced replica reaches the shared dense-DP bucket 141.8 ms later.
   Only 28.3 ms of that difference appears as additional GPU-active union;
   113.4 ms appears as additional GPU-inactive spacing. The reduced ranks
   also have 21.7% more stream-wait calls and 39.3% more aggregate graph-launch
   API time despite replaying the same 280 graphs.

This directly shows equal per-GPU routed work, finer GEMM fragmentation, and
more launch/dependency spacing. Calling fragmentation the cause of every
inactive microsecond would be stronger than the trace alone supports: the
inactive metric also includes host and inter-stream dependency stalls, and
EP4 native model-EP residency is 19.8 ms higher. The controlled evidence does
support the narrower conclusion that proportional batch allocation is not a
runtime-parity condition because the reduced EP4 graph executes the same
nominal work at a much finer expert granularity.

## Routing Imbalance

The fixed-bias A/B changes routing while preserving model, batch allocation,
topology, graph mode, and NEP implementation.

| Metric | Balanced | Fixed bias |
| --- | ---: | ---: |
| Full EP8 per-dispatch routed-load CV | 0.407% | 41.652% |
| Reduced EP4 per-dispatch routed-load CV | 0.397% | 26.369% |
| Balanced/biased healthy clean iteration | 952.233 ms | 1141.233 ms |
| Balanced/biased NEP clean iteration | 1018.833 ms | 1263.533 ms |
| Full-owner latency parity | 93.463% | 90.321% |
| Full-rank dense-DP launch | 875.948 ms | 1075.635 ms |
| Reduced-rank dense-DP launch | 1017.724 ms | 1366.701 ms |
| Reduced-versus-full arrival lag | 141.777 ms | 291.066 ms |
| Early full-rank dense-DP residency | 144.484 ms | 293.642 ms |
| Late reduced-rank dense-DP residency | 2.742 ms | 2.632 ms |
| Rank-0 dense-DP/model-EP co-residency | 6.256 ms | 293.898 ms |
| Healthy rank-0 native model-EP residency | 114.633 ms | 273.624 ms |
| NEP rank-0 native model-EP residency | 138.656 ms | 568.962 ms |

The matched 176,160,768-element dense-DP kernel is
`ncclDevKernel_AllReduce_Sum_f32_RING_LL`. On rank 0 it launches 32 blocks of
544 threads on CUDA stream 32 (`0.210526` blocks/SM), so it is an SM-resident
NCCL kernel rather than a copy-engine transfer. Its process group is
`[0, 2, 4, 6, 8, 10]`.

In the balanced case, full ranks launch this kernel around 876 ms and remain
resident for about 144 ms; reduced ranks launch around 1018 ms and run for
about 2.7 ms. Both groups finish around 1020.4 ms. Under fixed bias, the same
signature expands to 294 ms on the early full ranks and remains 2.6 ms on the
late reduced ranks; all participants finish around 1369.3 ms. This is direct
participant-wait evidence, not an estimate from iteration timing.

The full EP8 replica executes exactly 2,016 native model-EP collectives in
every case. Aggregate payload is also identical: 1,344 BF16 records carry
177,570,054,144 input and output elements, and 672 FP32 records carry
33,030,144 input and output elements. Nevertheless, biased NEP increases
rank-0 native model-EP residency by 295.338 ms versus biased healthy. That is
within 1.440 ms of the 293.898 ms dense-DP/model-EP co-residency. Non-NCCL
kernel residency is essentially unchanged at 659.144 ms healthy versus
655.791 ms NEP.

The evidence supports both parts of the routing claim:

- Imbalance worsens NEP. It adds 149.290 ms to the reduced-versus-full arrival
  lag, adds 149.158 ms to early-participant dense-DP residency, and creates
  almost 294 ms of overlap between that waiting SM-based kernel and native
  model-EP work.
- Imbalance is not the cause of the balanced proportional regression. The
  balanced router already has only about 0.4% per-dispatch CV and about
  `1.001x` aggregate max/mean, yet reduced ranks are still 141.8 ms late and
  owner parity is only 93.46%.

## Interpretation Boundary

NCCL kernel duration on an early rank includes participant wait, so it must
not be read as transport time. Here that property is useful: the early and
late kernels finishing together identifies the wait directly. The trace also
shows co-residency, unchanged operation count/payload, and unchanged
non-NCCL residency; together those observations support contention as the
reason native model-EP residency inflates in biased NEP.
