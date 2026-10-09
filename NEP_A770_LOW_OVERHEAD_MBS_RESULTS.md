# a770 MBS1/MBS2: low-overhead operation profiling

Date: 2026-09-11. Job 3007029 completed, exit 0, elapsed 27m55s.

## Result and measurement contract

All numbers below use iterations **40-49**, after warmup, for both healthy and
NEP. Nothing is corrected by subtracting profiler overhead. Parity means healthy
iteration time divided by NEP owner iteration time, not equal-global-batch
throughput. The six full-replica NEP owners are ranks 0-5.

| MBS | Healthy mean +/- sample SD, ms | NEP mean +/- sample SD, ms | Parity | NEP extra time |
| --- | --- | --- | --- | --- |
| 1 | 425.61 +/- 0.95 | 481.69 +/- 4.13 | 88.36% | 56.08 ms / 13.18% |
| 2 | 702.83 +/- 5.46 | 733.95 +/- 3.72 | 95.76% | 31.12 ms / 4.43% |

This is **targeted CPU/CUDA-event operation profiling**, not Nsight, Kineto, or a
complete CUPTI kernel trace. Full-kernel modes tested before this retained too
much differential instrumentation overhead. These traces retain grouped expert
forward/backward, NEP gradient staging, parameter redistribution, optimizer and
parameter-copy ranges on every rank. They cannot provide exact per-collective
busy/exposed time. Operation elapsed ranges include stream dependency and host
submission gaps; they must not be described as pure GEMM time.

## Profiler calibration

Each configuration has an independent, fresh-process unrecorded control on the
same allocation, plus unrecorded windows 29-38 and 51-60 in the profiled process.
The arithmetic average of these surrounding windows is a drift diagnostic, not
a corrected measurement or proof that drift was linear.

| MBS | Recorded parity | Before recording | After recording | Surrounding-window parity | Fresh-control parity |
| --- | --- | --- | --- | --- | --- |
| 1 | 88.36% | 87.88% | 88.71% | 88.29% | 89.78% |
| 2 | 95.76% | 96.17% | 95.27% | 95.72% | 95.54% |

Relative to the surrounding-window mean, recording adds 1.12% healthy / 1.04%
NEP for MBS1; MBS2 is -0.06% / -0.10%, indistinguishable from runtime drift at
this resolution. Recorded parity differs from that diagnostic by +0.0657 and
+0.0379 percentage points. Relative to the fresh controls, changes are -0.17% /
+1.44% for MBS1 and -0.83% / -1.05% for MBS2. Thus we do not claim zero overhead
or a universal sub-one-point bias bound: MBS1 has a 1.42-point fresh-control
discrepancy. Nevertheless the MBS2 advantage is present in every window,
including independent controls (5.76 points), not only recorded iterations
(7.40 points). Old full-kernel profiles are not mixed into the numbers below.

## Configuration and isolation

- Original a770m_4b_moe_215b: H1280, expert F768, 128 experts, top-k6,
  15 MoE layers, shared-expert F1536, sequence length8192.
- TP/CP/ETP/PP all1. Healthy EP8/8, NEP EP8/6. GA3 in both replicas.
- MBS1: healthy GBS48, NEP GBS42; full/reduced replica batches24/18.
- MBS2: healthy GBS96, NEP GBS84; full/reduced replica batches48/36.
  Reduced batch remains proportional. MBS and GBS double; GA and expert count
  do not change.
- DistOpt, overlapping grad reduce/param gather, Flex/HybridEP, non-divisible
  expert ownership, eight expert gradient groups. Existing CUDA graph scopes
  retained. Random forced balancing retained, not a fixed exact routing map.
- Same four GB200 nodes lyris[0266,0268-0270], segment4. Replicas contained in
  NVLink domains; software MNNVL cliques1000/1001 isolate cross-replica traffic
  onto RDMA. CPU binding none; OMP_NUM_THREADS=1.
- Eight cases: MBS1 control NEP/healthy, MBS1 recorded NEP/healthy, MBS2 recorded
  healthy/NEP, MBS2 control healthy/NEP. Case order is not randomized; runtime
  drift and routing variation remain possible. No energy or placement change
  was introduced between these cases.
- Trusted training checkout unchanged. nonuniform_ep.py SHA256 remains
  4676f164a95255bdcc7e2531a06ca92c41a2583bf6f9a7a85b3b74b64f26f2d1.

## Where the measured time difference changes

The following are **disjoint wall-clock intervals**, averaged across owners0-5
and measured iterations. Boundaries are CUDA-event timestamps surrounding the
first/last grouped expert operations of each microbatch. These intervals include
intervening non-expert compute, communication and gaps. They are not sums of
kernel durations and not exclusive attribution to the named operation.

| Interval | MBS1 NEP minus healthy, ms | MBS2 NEP minus healthy, ms |
| --- | ---: | ---: |
| Iteration start through final expert forward in first microbatch | 28.426 | 19.788 |
| Subsequent work through final expert backward | 11.381 | 8.748 |
| Final expert backward end to optimizer GPU-range start | 8.577 | 1.194 |
| Optimizer GPU range | 2.072 | 1.472 |
| Optimizer GPU-range end to iteration end | 5.629 | -0.114 |
| **Total** | **56.085** | **31.089** |

The ~0.03ms difference from logged deltas reflects rank averaging and different
timing boundaries. All per-rank temporal partitions telescope exactly to their
recorded iteration range; there is no unattributed residual in this *temporal*
partition. That does not imply every kernel's causal contribution is identified.

Of the ~25ms absolute gap reduction, 8.64ms is in the first-forward region,
13.72ms is in the post-expert tail, and 2.63ms is in the intervening region.
The first two microbatch expert-backward spans are essentially identical to
healthy at both MBS settings (deltas +0.033/+0.064ms at MBS1 and +0.020/-0.016ms
at MBS2). The last microbatch backward-span penalty persists: 4.316 versus
4.545ms. The improvement is not a uniform acceleration of NEP backward.

### 1. Parameter delivery becomes less submission-limited

DistOpt redistributes expert parameters in the first forward, once per expert
group, not at the end of this iteration. Both MBS settings have eight such
calls and the same parameter payload.

| Parameter redistribution measurement | MBS1 | MBS2 |
| --- | ---: | ---: |
| Full owners: summed CPU wall time, ms | 19.443 | 16.655 |
| Full owners: GPU operation-range union, ms | 16.703 | 11.615 |
| Reduced owners: summed CPU wall time, ms | 19.735 | 19.081 |
| Reduced owners: GPU operation-range union, ms | 14.059 | 5.243 |
| Full owners: CPU call finished before GPU range began | 1.7% | 54.6% |
| Reduced owners: CPU call finished before GPU range began | 0% | 40.0% |

The reduced owners spend nearly the same CPU time on these calls, but much more
of the submission is completed before the GPU arrives at MBS2. This directly
shows improved launch headroom, not less parameter traffic or faster CPU work.
The range reduction includes eliminated launch/dependency spacing and is not
a measurement of increased network bandwidth. It supports the first-forward
improvement, but cannot assign all 8.64ms solely to Scatter kernels.

### 2. The optimizer/finalization tail has more launch headroom

Full-owner NEP gradient staging CPU time stays about constant: 9.733 to 9.935ms.
The mean GPU start lag behind its CPU submission grows from 14.287 to 59.723ms.
Optimizer CPU duration changes only from 17.477 to 16.520ms, while its GPU start
lag grows from 22.483 to 87.786ms. All owner optimizer calls are submitted before
their GPU range starts at MBS2, versus 71.7% at MBS1. Parameter copies similarly
go from 88.3% to 100%. These timings establish that substantially more host
submission runs ahead of the queued GPU work at MBS2.

The actual post-expert tail is healthy12.986 / NEP29.264ms at MBS1, and
healthy20.577 / NEP23.129ms at MBS2. Its relative penalty shrinks by13.72ms, but
**not all of that is NEP becoming faster**: NEP shortens by6.13ms and healthy's
remaining work lengthens by7.59ms with larger batches. The targeted ranges do
not separate the latter into remaining model work versus collective waiting.
Likewise, optimizer-end to iteration-end is not automatically Scatter: timing
barriers and finalization remain there. No exclusive NCCL-wait cause is claimed
from traces that do not record individual NCCL kernels.

### 3. Amortization also helps; GEMM efficiency is not established as the cause

If MBS1's 56.08ms gap merely stayed fixed at MBS2's healthy time, parity would
already rise to about92.61%. The observed95.76% additionally requires the
absolute gap to shrink to31.12ms, located above. Thus this is both amortization
and an improved execution schedule, not amortization alone.

Expert backward operation-range sums are healthy23.291 / full-owner23.350 /
reduced25.685ms at MBS1, versus36.734 / 36.883 / 37.963ms at MBS2. The reduced
excess shrinks by only about1.30ms. Nearly all backward CPU calls already
finish before their GPU operation starts at *both* MBS settings. The reduced
backward CPU time actually rises78.325 to80.646ms. These observations do not
support attributing the ~25ms improvement to reduced backward launch overhead
or a large GEMM-efficiency recovery. Pure GEMM FLOP/s is not measured here.

## Validation, reproduction and artifacts

- Correctness smoke3006965: bitwise-identical outputs, input gradients and
  parameter gradients with and without event wrappers. No production hooks or
  optimizer semantics changed.
- All60 rank files validated: each of ten iterations has90 expert forward and
  90 expert backward operations, matched CPU/GPU range counts, expected stage
  counts, nonnegative durations, and clock calibration uncertainty<200us.
- CUDA events are preallocated/materialized outside measurement. No event
  synchronization, elapsed-time reads, CUPTI subscriber or trace writes occur
  inside the measured window. GPU operation boundaries do not certify occupancy
  of unrelated/helper streams; do not use them to compute total compute overlap.
- Training completed60 iterations in all eight cases. Batch exit0. Shutdown
  TCPStore/heartbeat warnings occurred after training; no case failed.
- Launch: `CUPTI_EVENTS_ONLY=1` through `run_a770_cupti_calibrated.sh`.
- Analyze: `python3 scripts/nonuniform/analyze_cuda_event_mbs_pair.py 3007029`.
- Root: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3007029`.
  Recorded traces: `mbs{1,2}/cupti/ep8/scaling_ladder__a770m_4b_moe_215b/{healthy,nep}/3007029/cupti_profile/rank-N.json`.
  Primary result: `cuda_event_analysis/summary.json` and `validation.log`.
- The launcher's older `comparison_3007029.json` files select post-window51-60.
  They are **not** the measured-window result used in this report.

Conclusion: MBS2's better parity is reproduced with low differential recording
overhead and independent controls. The traces locate the improvement chiefly
in first-forward parameter delivery and the post-expert tail, and show more
host launch work queued ahead of GPU execution. Exact attribution of each saved
millisecond to a specific collective, pure GEMM efficiency, and zero profiler
bias are not established and are not claimed.
