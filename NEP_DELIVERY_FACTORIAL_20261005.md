# Delivery Scheduling Attribution

## Question

Which parts of parameter delivery advance forward completion, and why? This
is not a search for a deliberately poor baseline. The completed MBS ablation
is paused. Preserve the previous branch at2280f1911.

## Independent Controls

Use the previously tested actual15-MoE a8b configuration at sequence4096,
H4608/latent1152/F3072,E512/topk6,MBS2/GA8,TP2/CP1,DistOpt,HybridEP,native
graphs. HealthyEP16/16 has local batches128/128;NEPEP16/12 has128/96.

| Cell | Expert AG lookahead | NEP redistribution release |
| --- | --- | --- |
| H0 | None | Healthy,not applicable |
| H1P | One bucket | Healthy,not applicable |
| Q00 | None | At consumer demand |
| Q01 | None | Immediately after AG submission,with device dependency |
| Q10 | One bucket | At consumer demand |
| Q11 | One bucket | Immediately after AG submission,with device dependency |

No-lookahead retains any native initial launch,only suppressing next-bucket
prefetch. It does not inject delays,barriers,change transfer bytes,or allocate
less memory. Early release without lookahead may have little effect;that is
a useful interaction control,not a promised slowdown.

Only production change is allowing existing NEP prefetch depth0 and consumer
release with that depth. Healthy uses native finish_param_sync with its
existing skip_next_bucket_dispatch argument,only for expert buckets. No edits
to native DDP/optimizer. Dense schedules and all backward code are unchanged.

Bucket layouts stay fixed within healthy and within NEP (existing actual
dense3/3,expert5/3);do not interpret cross-topology timing as an iso-bucket
protocol comparison. Difference-in-differences describes the gain of the
complete NEP versus healthy delivery schedules,not pure wire cost.

## Validation And Timing

Four-GPU numerical tests first:all four policies,fresh parameters,buffer reuse,
force-sync,AG/RS,modelEP and independent compute. Also test healthy no-lookahead
against native parameter values. Then EP8/6 four-layer training with all four
policies and both healthy controls. Full workload follows only after these pass.

Full study:one GB200 or GB300 eight-node segment4 allocation,each replica fully in its
NVLink domain;distinct NCCL cliques and verified RDMA between replicas. GB200
is also eligible for4K:previous peak reserved146GiB plus about8GiB nonallocator
memory leaves headroom on GB200. Do not propagate this allowance to MBS4/8K.
Six cells,two reversed-order repeats,profile and separately authorized timing
companion per cell:24cases total. Keep35iterations,measure25-30 after24warmup.
All-rank AG/RS/Gather/redistribution payloads,host submission,GPU completion,
consumer demand/ready and Nsight kernels remain captured,event-trace=false.
Fail closed on weights,routing,payload,placement,memory or missing records.

## Attribution

Report both main effects and their interaction:Q00-Q10 andQ01-Q11 for AG
lookahead;Q00-Q01 andQ10-Q11 for early redistribution. Compare H0-H1P to
separate native prefetch benefit. Keep profile and companion timing separate.

For every first-forward expert bucket and rank,match AG completion,
redistribution submission/completion,and demand/ready boundaries. Explain how
much earlier readiness advances actual forward completion,including follower
or modelEP dependencies. Distinguish initial bucket from later buckets.
Use Nsight kernels for kernel residency,not elapsed CUDA-event spans. Do not
add overlapping waits across ranks or buckets into iteration savings.

Deliverable:matched timing/parity and phase table;per-bucket delivery timeline;
effects/interaction chart with repeat variability. Any unexplained residual
is reported rather than assigned to host overhead by subtraction.

## Progress

- Gate3240405 failed in the no-lookahead smoke assertion before training.
  All parameter-value checks passed. Its delayed instance method had captured
  the uninstrumented class method,omitting the first redistribution range;
  the assertion therefore compared nextAG to the second delivery. Install the
  tracer before capturing the method,and require all three delivery ranges.
  No training implementation change. Healthy controls now both install the
  same wrapper so wrapper overhead is matched. Rerun the entire small gate.
- Gate3240472 COMPLETED0:0 in20m05s onGB2004nodes/segment2. All four
  numerical policies and healthy native-value controls pass. All six training
  cases complete25iterations;all88rank profile triples exist. All four NEP
  training/profile/transfer audits pass. No hang,OOM or invalid parameter value.
- Full sweep order:Q00,H0,Q01,H1P,Q10,Q11 in repeat0;reverse in repeat1.
  Repeat0 profile then companion;repeat1 companion then profile. Exact model,
  bucket layout,payload,work allocation and profiler configuration stay fixed.

## Completed Timing Results

Job3240633 COMPLETED0:0 in1h39m20s onGB2008nodes/segment4. Replica0 used
lyris0027,0030,0032,0033 in nvlblk02;replica1 used0075,0078,0079,0086 in
nvlblk05. All24cases pass;all352rank profile triples exist. Cross-replica
optimizer groups use RDMA;local redistribution groups use P2P. No hang/OOM.
First actual case peak allocated127.733GiB,min end-free31.424GiB.

Means of the two reversed-order repeats;each uses matched iterations25-30.
All NEP parity below uses the same prefetched healthy reference H1P. H0 is a
diagnostic for native AG prefetch,not a changing denominator for NEP parity.

| Cell | Companion iteration ms | Companion parity | Profiled iteration ms | Profiled parity | Profiled first forward ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| H0 | 1817.065 | -- | 1861.109 | -- | 135.568 |
| H1P | 1788.947 | 100% | 1834.672 | 100% | 104.492 |
| Q00 | 1929.736 | 92.704% | 1996.103 | 91.913% | 169.948 |
| Q01 | 1931.943 | 92.598% | 1990.190 | 92.186% | 170.002 |
| Q10 | 1899.130 | 94.198% | 1950.910 | 94.042% | 136.484 |
| Q11 | 1890.833 | 94.612% | 1954.725 | 93.858% | 131.464 |

- AG prefetch with consumer release saves33.463ms of profiled first forward.
  Native healthy prefetch saves31.076ms;most of the gain is ordinary AG
  pipelining,not a NEP-specific transfer optimization.
- Early redistribution with prefetch saves another5.020ms of first forward
  (4.950/5.091ms across repeats). Without prefetch it saves essentially none:
  mean -0.054ms,opposite signs across repeats. This is the interaction effect.
- Complete pipeline saves38.484ms of profiled first forward and38.903ms in
  separate timing companions. Companion parity increases1.907points.
- Early release reduces summed owner consumer waits32.030->22.871ms,but
  first forward improves only5.020ms;about4.139ms is offset elsewhere in that
  same phase. The kernel-level accounting below closes this difference.
- Profiler intrusion remains relevant:Q10->Q11 companion iteration improves
  by8.297ms (11.712/4.883ms),while profiled iteration worsens by3.815ms. The
  first-forward gain repeats in both profiled runs,but profiled whole-iteration
  timing must not stand in for unprofiled timing. No post-window substitution.

Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3240633.
Runtime checkpoint150021cbb;production defaults and prior branch unchanged.

## Why Local Waiting Savings Are Not All Latency Savings

For Q10->Q11,the following first-forward intervals are disjoint. Checked
parameter-wait/native-graph intersections in every owner/iteration;none exceeds
50us of trace-boundary tolerance. Negative means early release is faster.

| First-forward component delta,ms | Repeat0 | Repeat1 | Mean |
| --- | ---: | ---: | ---: |
| Parameter-ready wait | -9.130 | -9.190 | -9.160 |
| Native graph execution | +4.410 | +4.110 | +4.260 |
| Other first-forward intervals | -0.230 | -0.011 | -0.120 |
| Total first forward | -4.950 | -5.091 | -5.020 |

The native-graph partition shifts mainly into model compute(+2.275ms) and
model-EP(+2.064ms);other graph categories partly offset these increases.
These are nonoverlapping categories within the native graph,not independent
collective residencies added to compute. "Graph idle" means no native graph
kernel,not necessarily an idle GPU:other-stream redistribution may be active.

All16full-replica ranks have the same ordered105model-EP kernels per first
forward. With consumer release,followers are the latest kernel starters for
1.1%/6.7% of matched calls;with early release,72.7%/46.7%. Owner model-EP
residency before the latest peer kernel starts increases2.698/1.345ms. This
supports additional participant-arrival skew,not a claim that every extra
millisecond is wire transfer. Cross-host clock offset was not independently
measured;the peer-start diagnostic is not a pure-wait measurement.

Interpretation:AG lookahead creates the opportunity;early redistribution uses
it to deliver later buckets before demand. That overlap is not free in the
profiled workload:some savings are offset by longer native execution. The
experiment identifies the schedule-induced tradeoff,but does not isolate
SM versus memory-bandwidth versus NVLink contention individually. Nor do we
use profiled kernel costs as an exact decomposition of companion timings.

The initial bucket remains at about22ms of consumer wait across these controls.
These schedules do not move its initial launch boundary;this is not proof that
startup delivery is fundamentally irreducible.

## Published Artifacts

Figures, numerical tables, and this report:
https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_delivery_factorial_3240633/

All 352 rank kernel caches and the completed analysis are retained under the
job's `delivery_factorial` directory. The `analysis_source` directory archives
the analysis scripts and their local helper dependencies. Run the entry points
with that directory on `PYTHONPATH`; the original training data stays in the
job directory on Lustre. SVG and PNG figures were visually checked. Production
defaults are unchanged; no additional benchmark job is needed for this report.
