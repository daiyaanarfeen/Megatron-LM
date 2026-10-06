# Expert Bucket Sweep, 2026-10-06

Isolated from `nep-high-parity-delivery-20261005` at `7c23ac8e2`.
No training runtime changes. Revert by using that parent branch.

## Question

Does reducing the NEP expert bucket size shorten first-forward parameter
delivery and reduce final-backward interference with native compute/model-EP?
Test 3 (reference), 5, and 6 expert buckets using the existing environment
setting. Do not change dense buckets or native healthy bucketing.

## Fixed Controls

- Actual a8b latent hybrid/MTP saved memory-limited architecture, 15 MoE layers;
  H4608, latent1152, expert F3072, E512, top-k6.
- Healthy EP16/16, NEP EP16/12; TP2, CP1, ETP1, PP1.
- MBS2, GA8, sequence8192. Local BS128/128 healthy, 128/96 NEP.
- BF16 distributed optimizer, Flex/HybridEP, native MoE CUDA graphs.
- Exact randomized balanced routing, compact expert placement, ascending
  transfer order, one transfer stream, early delivery, prefetch depth1.
- Eight GB200 or GB300 nodes, four GPUs per node, segment4. A replica must fit
  in one NVL domain; clique isolation retains cross-replica RDMA transport.
- 35 iterations; measured25-30 after24 warmup. All-rank AG/RS, NEP redistribution,
  gradient Gather, phase, consumer readiness, and kernel traces.
- Nsight CUDA event tracing stays disabled to avoid its known false-dependency
  hang; custom GPU completion events remain enabled.

## Run Order

Each case has profiling and the previously authorized timing companion.
Repeat0: B3, H8K, B6, B5 (profile first).
Repeat1: B5, B6, H8K, B3 (companion first).
Sixteen cases, 15-minute case timeout, two-hour allocation cap.

The audit requires identical initialized weights, routing, dense buckets,
ordered expert slots/sizes, total expert AG/RS payloads, and destination-exact
redistribution bytes. Only expert bucket boundaries and expert collective
counts may differ. Numerical smoke tests run before the sweep.

## Acceptance

Report owner iteration parity against the fresh healthy control separately
for matched profiled iterations and companions. Report first forward, other
forward, nonfinal backward, final backward, and outside-F/B deltas. Check all
ranks, actual bucket sizes, memory, submission/completion, and native EP/TP
kernel durations. Overlapping residency differences are not additive causes.
Do not accept a first-forward gain that hides a backward/iteration regression.

## Preflight

Three CPU control-normalization tests pass. Python AST, shell syntax, git
whitespace, and unchanged `megatron/` runtime checks pass. Import order fixed
with `uv run isort`; formatting applied to modified Python files.

## Results

Job 3250917 completed successfully on GB300 in 1:25:43; all 16 cases passed.
Replicas occupied nvlblk31/32; all cross-replica optimizer pairs used RDMA.
Runtime checkpoint: `2e3324680`. Training implementation is unchanged.
Earlier GB200 job 3250818 completed its first 35-iteration case, but stopped at
the unchanged 16/8 GiB memory-headroom gates. It was not an OOM and is excluded
from the comparison below.

Owners 0-11, steps 25-30, two reversed-order repeats. Companion means refer to
the same numbered iterations of the separately authorized timing companions,
not post-profile windows. Parity is matched healthy time divided by NEP time.

| Configuration | Companion ms | Companion parity | Profiled ms | Profiled parity |
| --- | ---: | ---: | ---: | ---: |
| Healthy, native 5 expert buckets | 3148.90 | 100% | 3195.76 | 100% |
| NEP 3 | 3228.39 | 97.54% | 3283.01 | 97.34% |
| NEP 5 | 3209.53 | 98.11% | 3256.90 | 98.12% |
| NEP 6 | 3211.07 | 98.06% | 3259.47 | 98.05% |

Five buckets saves 18.87 ms of companion iteration time and gains 0.57 percentage
points of parity versus 3. Six is only 1.54 ms slower than 5; do not claim that
their difference is statistically resolved by two repeats. Both improve
over 3 in both repeats. The default remains 3 outside this experiment branch.

Profiled phase overheads versus healthy (ms; additive phase partition):

| NEP buckets | First F | Other F | Nonfinal B | Final B | Outside F/B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 25.33 | 1.06 | 1.41 | 38.65 | 20.77 |
| 5 | 19.33 | 0.08 | -1.08 | 31.20 | 11.61 |
| 6 | 15.80 | 0.64 | 2.57 | 37.83 | 6.88 |

Profiling adds a differential NEP penalty of 7.76 ms for 3 buckets, 0.51 ms
for 5, and 1.54 ms for 6. Thus the 26.11 ms profiled 3-to-5 gain must not be
substituted for the 18.87 ms companion gain.

Direct trace evidence (means across owners/steps/repeats):

| Configuration | First AG input MB | AG pending after preceding graph ms | Final grad-sync drain ms | Expert RS pending after backward ms |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 594.54 | 6.97 | 8.26 | 8.26 |
| NEP 3 | 1509.95 | 17.27 | 15.38 | 9.78 |
| NEP 5 | 905.97 | 10.51 | 9.51 | 2.32 |
| NEP 6 | 603.98 | 7.09 | 5.84 | 0.00 |

Total expert AG/RS bytes are identical across NEP counts. Scaling and
optimizer means remain effectively unchanged. Peak allocated memory falls
from 135.53 GiB to 130.99 GiB for both 5 and 6 buckets.

## Remaining Backward Interference

The 31.20 ms final-backward excess with 5 buckets remains the largest phase
overhead. Detailed first-repeat kernel analysis covers all 16 full-replica
ranks, six steps, all three NEP counts and healthy. Ordinary backward graph
execution is essentially unchanged. Final backward has native RMSNorm launch
gaps and longer TP/HybridEP waits; graph submissions are already queued at
least 117 ms ahead, ruling out late host graph submission in those samples.

A concrete old-trace chain, also showing the same gap signature in newer
traces with Nsight CUDA event tracing disabled: rank 4 pauses 13.9 ms before
RMSNorm, its TP partner waits, and rank 0 subsequently spends 14.6 ms in a
HybridEP sync while the last participant spends 0.005 ms. Those are propagated
waits, not separate additive costs or NVLink transfer times.

Leading hypothesis: cooperative RMSNorm launch admission with insufficient
SM headroom for simultaneous Gather/RS. The trace reports 152 SMs, a 360-CTA
norm grid, and concurrent 32-CTA Gather plus 24-CTA RS. Backward norm margin
is 32. TE's general norm launcher uses occupancy-based cooperative grids
when a row spans multiple CTAs. Healthy can also experience such gaps, so
changing the margin must be tested on both healthy and NEP; no guaranteed
parity gain is claimed. No margin or scheduling implementation change was
made in this study.

Source: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu

Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3250917`.
`expert-bucket-summary.json` contains matched repeat statistics and operation
ranges; `work-controls-audit.json` contains transport checks. Per-case
`actual-validation.json` records exact startup/routing/payload gates.
`backward_interference_B3`, `B5`, and `B6` contain detailed first-repeat
native-kernel analysis. Operation elapsed ranges include waiting/queueing;
overlapping collective residency is not summed as causal e2e overhead.
