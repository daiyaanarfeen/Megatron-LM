# Reduced-replica arrival audit

Existing traces only; no training changes or new jobs. Job3066391 prefetch
and job3073618 full_nep, iterations40-49. Six full owners0-5 versus six
reduced owners8-13; timestamps aligned by each file's integer origin_ns.
Each forward/backward microbatch contains30 expert linear calls. Measurements
below average all60 owner/iteration samples, not selected iterations.

## Where lateness accumulates

Job3066391 reduced-minus-full GPU boundary timestamps, milliseconds:

| Microbatch phase | Start skew | End skew |
|---|---:|---:|
| First forward | -3.880 | -6.386 |
| First backward | -5.836 | 16.669 |
| Second forward | 18.905 | 33.667 |
| Second backward | 34.039 | 60.537 |
| Third forward | 63.132 | 77.237 |
| Final backward | 77.673 | 101.133 |

The reduced replica exits first forward EARLIER. It accumulates most of its
lateness in ordinary microbatches before final gradient synchronization.
Healthy second-replica skew grows only from0ms at first forward to14.205ms
at final-backward start,16.711ms at final-backward end.

## Expert execution and CPU work

Job3066391, second microbatch (neither first-forward redistribution nor
final-backward gradient synchronization), milliseconds per rank:

| Measurement | NEP full | NEP reduced | Difference |
|---|---:|---:|---:|
| Forward expert-to-expert elapsed span | 80.641 | 95.403 | 14.762 |
| Summed forward grouped-expert active CPU time | 32.301 | 41.555 | 9.254 |
| Summed forward grouped-expert GPU event ranges | 32.844 | 42.092 | 9.248 |
| Forward time between those expert ranges | 47.798 | 53.311 | 5.513 |
| Backward expert-to-expert elapsed span | 92.127 | 118.625 | 26.498 |
| Summed backward grouped-expert active CPU time | 59.937 | 78.981 | 19.044 |
| Summed backward grouped-expert GPU event ranges | 18.794 | 33.517 | 14.723 |
| Backward time between those expert ranges | 73.332 | 85.108 | 11.776 |

CPU values use thread_cpu_us, not just wall elapsed. CPU and GPU values
overlap: do NOT add CPU time to elapsed span or claim every extra CPU
millisecond is exposed. Forward GPU range growth matches CPU scope growth;
rank0's average forward GPU-end minus CPU-return is -0.002ms, rank8 -0.007ms
(clock-scale uncertainty). Neither has a pre-submitted queue hiding those
whole forward calls. Backward has significant submission-ahead activity and
cannot be assigned wholly to exposed CPU time.

EP8 has64 real experts/GPU; EP6 has85-86. Same proportional tokens/GPU
therefore does not mean the same host work: grouped-expert CPU time grows
28.6% forward and31.8% backward, close to33.3% more real experts/GPU.
This supports a host-side expert-count penalty rather than demonstrating
an equivalent drop in GEMM hardware throughput. The event profiles do not
resolve individual TE helper/launch functions or prove exact per-expert costs.

## Independent profiling-run check

Job3073618 full profiles, second microbatch:

| Measurement | Full | Reduced |
|---|---:|---:|
| Forward span | 80.738 | 96.035 |
| Forward expert active CPU | 31.498 | 40.965 |
| Backward span | 91.771 | 115.908 |
| Backward expert active CPU | 59.427 | 77.325 |

Reduced starts final backward80.725ms late and ends100.182ms late.
The independent sparse run has56.379ms/83.127ms respectively: lateness
persists without broad profiling, but do not transfer broad-profile phase
attribution or exact magnitudes to the sparse/uninstrumented run.

## What these traces do not establish

These are CUDA-event OPERATION ranges, not complete kernel traces. They
omit HybridEP device polling/data kernels and intervening nonexpert kernels.
Thus they cannot fully split the26.5ms backward difference into GEMM service,
exposed host preparation, and model-EP peer waits. Nor can they prove that
all final-backward slowdown has the same cause. Any claim of a fully measured
GEMM-efficiency loss or exact CPU-critical-path attribution would overstate
the evidence. The measured result is increased ordinary expert-path CPU work
and accumulated ordinary-phase delay, preceding final RS readiness.

Sources: Lustre soft_clique_baseline/3066391/prefetch and3073618/full_nep,
cupti/ep8/scaling_ladder__a770m_4b_moe_215b/nep/<job>/cupti_profile/rank-*.json.
GPU range span = last end minus first start; CPU active time = sum of
thread_cpu_us for30 matching expert calls. Native TEGroupedLinear.forward
calls super().forward in the candidate runtime; no alternative NEP expert
GEMM implementation is inferred here.
