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
