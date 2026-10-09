# Cache CPU slot indices: a8b validation

## Scope and correctness

Branch `nep/slot-cache-20260917`, commit `e105bd46c`, based on compact parameter
transfer `d638286d7`. Six added runtime lines in token_dispatcher.py cache
immutable active-slot indices on CPU and select that cache for CPU token
counts. CUDA selection retains its existing path. No communication ordering,
expert placement, optimizer, bucketing or training-math changes.

One-node GPU test3091500 completed0:0 in1:41; all11tests passed. Tests cover
CPU/CUDA token counts, native dispatch, identity, padded/noncontiguous and
empty selections, repeated calls and a CPU-path guard against device-index
conversion. The same11tests passed in benchmark preflight. Worktree remains
clean; original compact/precompact/reference checkouts are untouched.

## Matched healthy / before / after: job3091300

COMPLETED0:0,17:17:07-17:43:44PDT,8GB200nodes,segment4. Replica0 entirely
nvlblk04, replica1 entirely nvlblk08. Healthy then pre-fix then fixed NEP,
45iterations each, all32/28/28rank operation profiles for iterations25-34.

Workload unchanged from the preceding attribution: depth15MoE, H4608,
latent1152,F3072,E512,top-k6,seq8192,MBS2,GA8,TP2,CP1. Healthy EP16/16,
GBS256,local batches128/128; NEP EP16/12,GBS224,local batches128/96.
DistOpt/Flex-HybridEP/graphs/recompute retained. Actual expert buckets5/8,
dense buckets3/3. Dynamic receive sizing and random-logit force balancing.

| Measured25-34 | Healthy | NEP before | NEP fixed |
|---|---:|---:|---:|
| Mean iteration, ms | 4878.83 | 5228.27 | 5201.53 |
| Iteration SD, ms | 18.91 | 12.88 | 12.97 |
| Parity | 100% | 93.32% | 93.80% |
| Extra iteration time | 0% | 7.16% | 6.61% |

The fix saves26.74ms in the measured window. The CPU postprocess removal
is much larger than this; do not count all removed CPU time as critical path.

### Additive owner phase overhead versus the SAME healthy run

| Phase | Before extra ms | Fixed extra ms |
|---|---:|---:|
| Startup + first forward | 82.790 | 77.564 |
| Other seven forwards | 45.144 | 5.746 |
| Earlier seven backwards | 51.658 | 15.764 |
| Final backward | 124.053 | 183.103 |
| Backward-to-optimizer drain | 26.845 | 26.248 |
| Optimizer + subsequent tail | 8.060 | 8.059 |
| Other transitions | 10.826 | 6.144 |
| Total extra, event anchors | 349.376 | 322.629 |

These are disjoint elapsed phases, not transfer service or pure compute.
Small differences from logged deltas come from different outer timing anchors.

The increased final-backward row is NOT slower end-to-end execution: owners
reach synchronization earlier, while reduced ranks do not gain as much.
Reduced-minus-owner final-B start skew409.365->470.948ms; end skew32.096->
31.727ms; reduced local final-B437.373->434.471ms. Thus the additional owner
final-B59.050ms equals -2.902 +61.584 -(-0.368)ms. Owner RS post-last-ready
sum110.362->110.491ms is unchanged; the longer ranges are mostly waiting.

### The intended synchronization is actually removed

Means over full owner ranks0-11 and measured iterations:

| Measurement | Healthy | Before | Fixed |
|---|---:|---:|---:|
| Postprocess CPU elapsed sum, ms/iteration | 0.926 | 78.605 | 4.194 |
| Postprocess GPU-event elapsed sum, ms/iteration | 0.570 | 11.279 | 0.574 |
| Expert CPU start minus dispatch GPU end, ms/call | -0.098 | +0.270 | -0.049 |
| Expert CPU calls beginning before dispatch GPU end | 85.6% | 0% | 66.8% |

Reduced ranks: postprocess CPU74.758->5.348ms, GPU range12.402->0.598ms,
expert CPU start offset+0.324->+0.034ms/call. Copies were removed on BOTH
replicas; some eliminated CPU waiting was already hidden behind GPU work.

### Remaining native expert work motivates the next experiment

Ordinary backward, microbatches1-6: full-owner healthy316.862ms versus
fixed318.970ms. Reduced fixed372.616ms versus healthy second-replica
349.838ms (including all16 healthy second-replica ranks).
Corresponding expert backward/recompute CPU thread time:73.408/73.486ms
on owners,82.185/101.249ms on second/reduced ranks. This is native TE scope
time, separate from dispatch postprocess, and may include driver waiting.
It does not prove every extra CPU millisecond is exposed or that pure GEMM
efficiency accounts for the remainder.

The separate all-rank arrival analysis DOES include all16 healthy second
replica ranks. Healthy final-B start/end skew346.192/20.783ms; fixed NEP
470.948/31.727ms. Fixed owner final-B extra183.103ms equals reduced local
extra69.292 +extra start skew124.756 -extra end skew10.945ms.

### Profiling distortion and validation limits

Post36-45 healthy4748.98ms, before5138.93ms, fixed5053.03ms; parity92.41%
and93.98%. The fix improves pre, measured and post windows, but by different
amounts:16.24,26.74,85.90ms. Do not replace the measured breakdown with the
post timings or claim one exact uninstrumented saving from this single run.

Native AG/RS, NEP Gather and redistribution peer sets match across all ranks
and ten measured steps. Elapsed collective ranges include waits and overlap;
they must not be added to the phase table. Random routing and floating-point
accumulation make benchmark loss equality an invalid exact-correctness test.
Selection tests, not short-run loss agreement, establish this patch's mapping
equivalence. No skipped/NaN training iterations observed.

Artifacts: Lustre `slurm_runs/soft_clique_baseline/3091300/slot_{before,after}_overhead_analysis/`:
`summary.json`, `delivery_chain.json`, `slot_cache_attribution.json`, rank caches.
Analysis scripts: `analyze_a8b_current_overheads.py`,
`analyze_a8b_delivery_chain.py`, `analyze_a8b_slot_cache.py`.

## Conditional work-per-expert experiment

Submitted3091845,8nodes,segment4,25min,GB200/GB300 eligible. Same healthy/fixed
comparison with sequence length8192->12288 and its matching position limit.
MBS2,GA8,H/F/E/top-k/depth, topology, buckets and profiling unchanged. This
increases tokens per expert invocation50% without increasing expert count or
number of expert host calls. It also increases attention/model-EP work, so
changes will be checked against each run's matched healthy baseline.

Four manifest transformation checks prove only sequence length and position
limit change relative to the current depth15workload. Initial ETA18:09PDT.
3091845started17:52:24PDT onGB300. A subsequent attempt to pin it toGB200
was rejected because it was already running. Therefore an additional fixed
8K healthy/NEP control onGB300 is required before attributing a parity change
to sequence length. Results pending; no parity or hiding improvement claimed.

## GB300 sequence-length results: 8K versus 12K

Both comparisons complete0:0:3091888(8K) in19:14 and3091845(12K) in21:10.
Both have all60rank profiles and45iterations/case. Matching manifests differ
only in sequence length, corresponding position limit and output directories.
The fixed NEP commit, MBS2/GA8/depth15/all other dimensions are identical.
Within each allocation each replica occupies one physical NVL domain and
the two replicas occupy different domains. Hardware is GB300 in BOTH rows;
do not compare the GB2008K row above directly against GB30012K.

| Seq | Healthy measured ms | NEP measured ms | Extra ms | Measured parity | Post-window parity |
|---|---:|---:|---:|---:|---:|
| 8192 | 4830.50 | 5099.21 | 268.71 | 94.73% | 94.69% |
| 12288 | 6179.55 | 6472.81 | 293.26 | 95.47% | 94.77% |

Measured SD healthy/NEP15.53/27.96ms at8K and15.05/25.81ms at12K.
Post36-45 healthy/NEP4730.55/4996.05ms at8K and6068.98/6403.91ms at12K.
The absolute overhead GREW in both windows. The apparent measured parity
improvement is mostly not reproduced post-profile: +0.739points measured
versus+0.084points post. This is a distortion/drift check, not a license to
replace measured timings with post timings. A robust hiding improvement has
NOT been demonstrated by the12K point.

### Additive full-owner overhead versus matched healthy

| Phase | 8K extra ms | 12K extra ms |
|---|---:|---:|
| Startup + first forward | 72.775 | 77.729 |
| Other seven forwards | -5.635 | 51.483 |
| Earlier seven backwards | -52.275 | 145.724 |
| Final backward | 216.105 | -14.392 |
| Backward-to-optimizer drain | 25.954 | 13.497 |
| Optimizer + subsequent tail | 8.115 | 7.373 |
| Other transitions | 3.695 | 11.818 |
| Total, event anchors | 268.735 | 293.232 |

The final-backward decrease alone is not an overhead reduction: more time
appears BEFORE that phase in12K. Summed owner RS arrival-wait ranges fall
997.235->364.651ms, but these overlap each other and do not represent removed
iteration time. Owner post-last-ready RS110.428->107.485ms is roughly stable.

All healthy second-replica ranks included, ordinary backward microbatches1-6:

| Measurement, ms/microbatch | 8K | 12K |
|---|---:|---:|
| Healthy second ordinary backward | 356.712 | 431.748 |
| NEP reduced ordinary backward | 372.629 | 451.121 |
| Healthy second expert backward/recompute CPU thread time | 78.404 | 84.029 |
| NEP reduced expert backward/recompute CPU thread time | 97.928 | 98.804 |
| NEP reduced expert backward/recompute GPU-event range sum | 80.962 | 80.718 |

Expert invocation counts are unchanged. More logical GPU work does not yet
produce longer expert operation ranges or a smaller reduced-replica latency
penalty here. These GPU ranges are not pure kernel busy time; native CPU
thread time can include driver waits. Do not equate the CPU delta with an
exact exposed contribution.

## Stronger work-per-expert point: 16K

Submitted3092074,GB300-only,8nodes,segment4,30min, seq16384 with the same
fixed code/MBS2/GA8/depth15. Started18:21:52PDT ontheia0202-0208,0210.
This is one further bounded increase because the12K gain is not robust
across timing windows. All other completed jobs are finished. Results pending.
