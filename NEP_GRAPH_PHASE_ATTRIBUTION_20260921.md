# Full-MoE graph residual overhead attribution

## Scope and evidence

Production unchanged at b798119e9 in
`.worktrees/nep-static-slot-capacity-20260921`. This investigation adds only
diagnostic launchers, profiling ranges, and analysis scripts.

Workload: depth-reduced a8b latent,15MoE layers,H4608,latent1152,F3072,E512,
topk6,sequence8192,MBS2,GA8,TP2,CP1,ETP1; full-MoE CUDA graphs, uniform static
routing, capacity1, Flex/HybridEP, BF16 distributed Adam. Healthy EP16/16,
NEP EP16/12. Replica-local batches128/128 versus128/96, GBS256 versus224.

Original same-allocation job3120733, measured25-34, owners0-11:

| Disjoint iteration phase | Healthy ms | NEP ms | Delta ms |
|---|---:|---:|---:|
| First forward |168.988|263.261|94.273|
| Other forwards combined |1093.962|1094.959|0.997|
| Nonfinal backwards combined |1632.020|1626.687|-5.333|
| Final backward |247.076|286.938|39.862|
| Outside forward/backward |49.894|69.966|20.072|
| Iteration |3191.940|3341.811|149.871|

Nsight graph-node diagnostics: NEP3120927,healthy3121008, measured25-27,
all28/32ranks. Every applicable rank has native AG/RS, NEP Gather/parameter
redistribution, PG membership, payload, host submission, and GPU completion.
Reduced owners' local Gather requires no network collective. Both cases
completed40steps. Healthy and NEP diagnostics used different GB200 allocations
with the same8node/segment4 resource shape and replica isolation. Do not
replace the original same-allocation totals with this diagnostic comparison.

Diagnostic owner means3274.483/3437.260ms (95.264% parity). First-forward
delta94.340ms and outside-F/B20.450ms reproduce the original closely;
final-backward delta32.144ms differs from39.862ms. Ordinary backward also
differs; do not transplant exact kernel-level milliseconds into the old run.

Instrumentation check using logged means, not substituted measurements:

| Window | Healthy ms | NEP ms |
|---|---:|---:|
| Pre-capture15-23 |3209.722|3347.689|
| Measured25-27 |3274.133|3436.900|
| Post-capture30-40 |3224.827|3383.409|

Measured/post parity95.264/95.313%. Profiling has overhead; its relative
distortion against this post window is small. This does not prove zero
distortion, identical allocations, or numerical equivalence of graph scopes.

## 1. First forward: future prefetch becomes a current-layer dependency

`NonuniformEPNCCLParamAndGradBucketGroup.finish_param_sync()` waits for the
current parameters, then calls the next bucket's `start_param_sync()` before
the current layer's graph is replayed. The latter submits native AG followed
by redistribution broadcasts, with correct AG-to-broadcast device dependencies.

However, the launcher also enables `NCCL_LAUNCH_ORDER_IMPLICIT=1`. NCCL2.30.4
adds per-device launch-order dependencies across communicators. A TP collective
captured in the subsequent compute graph is ordered after the already-submitted
future-parameter broadcast's **launch**. That broadcast cannot launch until its
AG and packing finish. Thus a correct local data dependency accidentally becomes
a dependency of otherwise independent current-layer computation.

Rank0,iteration25,firstMoE graph, Nsight timestamps in ms:

```
3573.088--3573.199  current-layer RMSNorm
3573.193--3580.352  next-bucket optimizer parameter AG
3580.371--3580.561  next-bucket packing
3580.569           next-bucket broadcast starts
3580.572           current-layer TP AG finally starts
```

The7.372ms graph gap ends2.912us after broadcast launch, not completion.
All seven affected first-forward graph boundaries show the same pattern,
with TP launches1.1-2.9us behind the corresponding broadcast launches.
The graph was submitted milliseconds earlier; this is not CPU starvation.

Followers receive several owner broadcasts sequentially. Onrank15 the first
MoE's equivalent TP delay is11.046ms, and subsequent affected layers wait
about13.7-13.9ms. Early owners subsequently wait in HybridEP for these followers:
rank0 first `ag_nvl_kernel` lasts5.879ms, rank15's0.104ms. The reduced replica's
firstMoE graph lacks these gaps. Its parameters need local copies, not the
full replica's owner-to-follower broadcast chain.

Disjoint graph-internal partition,rank0,three measured iterations, NEP-healthy:

| Component | Delta ms |
|---|---:|
| Gaps between this graph's kernels |51.544|
| HybridEP-only kernel residency, including peer waits |42.285|
| Native NCCL-only residency inside graphs |3.368|
| Compute-only residency |-0.482|
| Compute/HybridEP overlap |-0.002|
| Sum of graph-span differences |96.713|

These gaps are **not GPU idle**: future parameter traffic runs during them.
The table covers graph spans, not the complete first-forward scope.

**Fix direction:** remove the unintended cross-communicator launch dependency,
not the necessary AG-to-redistribution or parameter-use dependencies. First
perform a bounded correctness test with implicit ordering disabled; do not
change the production default blindly given historical multi-PG hangs. If
explicit ordering is required, change when future redistribution is submitted
so it cannot precede independent current compute in the global launch order.
Simply adding another CUDA stream or making broadcasts async will not remove
this dependency. This fix is proposed, not implemented or validated here.

Relevant code: `nonuniform_ep.py`1717,1771,1818; launcher
`tests/unit_tests/distributed/nep_param_transfer_cases.sh` sets the NCCL flag.
NCCL source matching the observed version:
[enqueue.cc v2.30.4-1](https://github.com/NVIDIA/nccl/blob/v2.30.4-1/src/enqueue.cc#L1628).

## 2. Final backward: Gather/staging gates subsequent TP launch; peers wait

The analogous chain is:

```
bucket gradients ready -> Gather -> owner-layout staging -> expert RS
                                                        |
                         subsequent layer's TP launch --+
                         (implicit launch-order dependency)
```

Rank0's eight affected subsequent backward graphs start their first TP AG
5.1-19.0us after the preceding expert RS launches. Their entry delays total
about14.04ms versus0.86ms healthy in iteration25. The CPU submitted an example
attention replay roughly320ms before GPU execution, excluding host starvation
as the cause of that example's delay.

There is also rank skew inside the full replica. At backward graph4,rank0's
HybridEP barrier runs6707.630-6709.788ms; rank10 arrives at6709.768ms and leaves
at6709.774ms (cross-rank clock offsets calibrated). Rank10 is an owner, not a
reduced-replica rank. Its preceding attention graph has slower native TP and
attention kernels while gradient communication is in flight. Therefore this
is not simply reduced-replica expert GEMM inefficiency.

Disjoint graph-internal partition,rank0,three measured iterations:

| Component | Healthy ms | NEP ms | Delta ms |
|---|---:|---:|---:|
| Native NCCL only |15.759|27.339|11.580|
| HybridEP only |11.505|23.783|12.279|
| Compute only |153.702|156.017|2.314|
| Compute/NCCL overlap |5.999|6.909|0.910|
| Compute/HybridEP overlap |0.009|0.005|-0.004|
| Gaps within graph span |15.785|6.940|-8.846|
| Graph spans total |202.760|220.993|18.233|

Graph entry delays are outside that table. Both the entry delays and the
internal peer waits matter. Summing all NCCL residency would double-count
overlap and would not explain the iteration delta.

**Fix direction:** first address launch ordering described above, retaining
Gather-to-RS dependencies and canonical RS bucket order. Only then assess the
remaining resource interference. The trace shows actual compute inflation,
not just collective waiting: e.g.rank10 attention backward6.411ms versus
rank15 4.781ms at one synchronization boundary, with the same full-replica
workload. This comparison alone cannot distinguish SM, memory-bandwidth, and
scheduling effects or prove all of that difference is caused by NEP.

A specific secondary hypothesis is insufficient normalization SM headroom:
GB200 has152SMs, backward RMSNorm launches408CTAs with167registers/thread,
and the configured SM margin is16; concurrent NCCL uses24/32CTAs. Norm launch
gaps align with NCCL activity, including in healthy. Testing a larger native
`NVTE_BWD_LAYERNORM_SM_MARGIN` is low complexity. It is **not yet a proven
fix**, and the graph trace's launch-type metadata is not sufficient to claim
cooperative admission is the sole cause. No claim that all39.862ms disappears
with an ordering or SM-margin change is justified before an intervention.

Relevant code: `nonuniform_ep.py`1336,1449,1851. The most important observed
remaining costs are device ordering and peer synchronization, not per-expert
Python submission or just raw RS transfer time.

## 3. Outside forward/backward: redundant scaling and larger optimizer shards

Original run,owners0-11,exact disjoint deltas:

| Outside-F/B segment | Delta ms |
|---|---:|
| Before first forward |2.376|
| Backward end to optimizer start |9.397|
| Optimizer step |8.802|
| After optimizer |-0.503|
| Inter-microbatch gaps |approximately0|
| Total |20.072|

New diagnostic identifies the operations:

- **Gradient scaling:** healthy14.066ms; NEP25.362ms. NEP calls base DDP
  scaling over physical dense+expert gradients (14.146ms), then separately
  scales logical optimizer-owner gradients (11.216ms). Optimizer construction
  selects owner proxies instead of physical expert parameters. After Gather
  and RS, scaling the physical expert copies is not needed for this Adam
  update. This is a concrete redundant pass, not intrinsic communication cost.
  Reduced ranks pay more physical scaling than full ranks, adding readiness skew.
- **Adam/update plus model copy:**13.286ms healthy versus17.423ms NEP owners.
  Copies are included (2.907 versus3.780ms); do not add them again. Owners
  represent42/43experts per layer versus32healthy, so more optimizer state
  and parameter traffic per owner is expected under the current ownership.
- **Gradient norm:**2.117 versus6.435ms. Rank0's scalar norm allreduce elapsed
  range is about3.76ms NEP versus0.22ms healthy. This includes peer waiting,
  not3.76ms to move four bytes. Do not disable this correctness collective.
- **Before first forward:** additional owner-gradient zeroing. Rank0 shows
  14additional approximately0.157-0.161ms vectorized zero kernels, consistent
  with the approximately2.4ms phase increase. Physical accumulation buffers
  still require reset; any owner-buffer reset removal must prove every used
  value is overwritten and that padding remains zero.
- Native finish-grad-sync is not an additional tail culprit in this diagnostic:
  8.356ms healthy versus6.869ms NEP owners.

**Minimal safe cleanup candidate:** in NEP DDP `scale_gradients`3262, when
using distributed optimizer, scale native dense buffers plus logical owner
buffers, omitting physical expert buffers. Preserve the existing non-distributed
optimizer path. Check optional gradient consumers before generalizing; this
benchmark's optimizer/grad-norm path uses proxies, but arbitrary external
gradient inspection is outside this demonstrated argument. Unit-test the
physical/owner distinction and compare gradients, gradnorm, and updated
parameters over multiple optimizer steps before accepting performance results.

The extra Adam payload is expected for this ownership design. The redundant
scaling is fixable; the gradient-norm wait may shrink when peer readiness
improves. Claiming the full20ms is unavoidable would be incorrect.

## Reproduction and limitations

All artifacts are under
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/`:

- `3120733/graph_residual_attribution.json`: original all-rank phase account.
- `3120927/graph_residual_attribution.json`: combined diagnostic phase account.
- `3120927/rank0_graph_kernel_partition.json`: disjoint graph partitions.
- `3120927/graph_attribution_nep/nsys/ep16/a8b_120b_latentmoe_1t/nep/3120927/`:
  NEP all-rank Nsight SQLite/reports and CUDA-event JSON.
- Analogous healthy path under3121008/graph_attribution_healthy.

Analysis scripts: `analyze_a8b_graph_attribution.py` and
`analyze_graph_kernel_dependencies.py`. The latter matches native graph
launch correlation IDs to kernels and uses interval unions/partitions.
Kernel detail above is rank0 plus explicitly identified peer examples;
phase accounting and required trace validation cover every rank.

No production fix was attempted. Padding-only numerical correctness remains
as previously validated; full-MoE graph numerical equivalence is still not
established by these timing runs. No optimizer/checkpoint scope was expanded.
