# Workload-Only Native MoE CUDA Graph Smoke

## Result

Native full-MoE graphs execute with Flex/HybridEP, distributed optimizer,
overlapped gradient reduction/parameter gathering, and NEP. No production
implementation changes. This is a small compatibility test, not full a8b
performance or complete numerical-equivalence validation.

Job3120243 completed0:0 on GB200 lyris0018/0162. Production checkout
`.worktrees/nep-slot-cache-20260917`, commit
`e105bd46c3966d916a6cb0276866eb85530e0529`, remained clean.

## Workload

One a8b-shaped MoE layer: H4608, latent1152, expert F3072,512experts,topk6,
shared F6144,squared-ReLU,BF16. TP2/ETP1/CP1/PP1,sequence1024,MBS2,GA2.
Healthy EP4/4,8GPUs,GBS16; NEP EP4/2,6GPUs,GBS12. Each replica occupies
one node; separate MNNVL cliques. This is a 2:1 EP difference, not EP16/12.

Graph-enabling workload flags:

```text
--cuda-graph-impl local
--cuda-graph-modules moe
--moe-expert-capacity-factor 1
--moe-pad-expert-input-to-capacity
--moe-router-force-uniform-routing
```

Control uses `--cuda-graph-modules moe_router` with identical padding/routing.
Exact uniform routing avoids variable expert splits in this synthetic test;
this does not validate arbitrary routing imbalance or nondivisible placement.
Selective MoE recomputation remains enabled. Launch script:
`scripts/nonuniform/run_nep_native_moe_graph_smoke.sh`.

## Measured Timings

Same allocation, twelve iterations/case; matched profiled log iterations8-10.
Capture starts before this stable subset; capture startup and export iterations
are excluded consistently. All-rank GPU-event operation profiling is enabled.

| Topology | Router-only graphs | Full-MoE graphs | Reduction |
| --- | ---: | ---: | ---: |
| Healthy EP4/4 | 83.733ms | 54.400ms | 35.0% |
| NEP EP4/2 | 150.233ms | 112.000ms | 25.4% |

Healthy/NEP parity is55.7%before and48.6%after: both speed up, but healthy
improves relatively more. Do not extrapolate this one-layer, tiny-token-count,
large-expert-payload test to full a8b or claim parity improved.

Pre-profile iterations4-6 average81.200/53.933/147.000/112.033ms in table
order. Measured versus pre-profile differences are3.12/0.87/2.20/-0.03%.
These include warmup/drift, so are not pure profiler-overhead estimates.

## Replay And Synchronization Evidence

All28rank profiles inspected. Over four captured iterations, full-MoE has
8forward and8backward MoE graph replays/rank and no eager expert forward
calls. Router-only has16eager expert-forward calls/rank, including recompute.
Native AG/RS remains present on every participating rank; NEP owner Gather
and parameter redistribution collectives remain present. Collective records
include PG ranks, payloads, host submission, and GPU event completion.

Rank0 full-MoE replay CPU ranges average0.128ms forward/0.117ms backward
healthy and0.119/0.119ms NEP. Router-only eager expert-forward CPU ranges
average4.469ms healthy and5.293ms NEP. Graph forward covers more than just
expert GEMMs; these numbers demonstrate removal of repeated eager expert
submission, not an additive attribution of the whole iteration speedup.

GPU event ranges measure elapsed operation intervals, not pure kernel service;
individual kernels inside captured graphs are not separately timed here.

## Numerical Limits

Healthy logged losses/gradient norms match between capture modes. NEP logged
loss differences reach0.3671%, gradient-norm differences0.005. Both complete
without NaNs or skipped steps. This does not establish gradient/parameter
equivalence; do not silently treat the difference as proven harmless rounding.
A separate tensor-level correctness comparison is needed before accepting
the configuration as fully validated.

## Artifacts And Failed Attempts

Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/native_moe_graph/3120243`.
Each case has `driver.log`, arguments, placement, transport logs, metadata,
and all-rank `cupti_profile/rank-*.json` GPU-event traces.

3120137 failed before training on missing attention-backend launch argument.
3120188 healthy cases passed, NEP failed pre-model from Slurm world-size
override. Both were launcher-only corrections, not runtime fixes. No jobs
remain active for this smoke.

## Nondivisible EP16 Graph Comparison

Job `3120733` completed all four cases, 40 iterations each, on the same GB200
allocation. Patch commit `b798119e9` on isolated branch
`nep/static-slot-capacity-20260921`; trusted `e105bd46` checkout unchanged.
Only runtime change is static capacity/count handling in `token_dispatcher.py`.
The accompanying unit tests cover logical capacity, dummy slots, count/cache
consistency, synthetic gradients, and native default behavior: 17 tests passed.

Workload: a8b latent depth-15-MoE variant, H4608, latent1152, expert F3072,
512 experts, top-k6, sequence8192, MBS2, GA8, TP2/CP1/ETP1, DistOpt,
Flex/HybridEP. Healthy EP16/16, GBS256, replica-local batch128/128; NEP
EP16/12, GBS224, replica-local128/96. Both graph modes use capacity1,
pad-to-capacity and exact uniform routing. They differ only in native graph
coverage (`moe_router` versus `moe`); Mamba/attention graphs remain enabled.
These routing/padding settings differ from the original dropless random-routing
benchmark, so this is a new matched control, not a direct comparison to old parity.

Segment4 places the full replica in nvlblk02 (lyris0025-0027,0032), the other
in nvlblk07 (lyris0111-0113,0123). Separate software cliques retained and NCCL
logs confirm cross-replica RDMA/GDRDMA. Original benchmark NCCL settings were
used, without the small-smoke PXN workaround. No OOM or further depth reduction.

### Matched Measured Iterations 25-34

Owner means use corresponding ranks0-11 in both topologies. Standard deviations
below are across the ten per-iteration owner-rank means, not across rank samples.
All120 rank profiles passed the common microbatch/replay/collective validation;
AG/RS records include groups, payloads, host submission and GPU completion.

| Native graph scope | Healthy ms, mean +/- SD | NEP ms, mean +/- SD | Owner parity | NEP extra ms |
| --- | ---: | ---: | ---: | ---: |
| Router only | 3630.405 +/- 5.584 | 3789.449 +/- 3.764 | 95.803% | 159.045 |
| Full MoE | 3191.940 +/- 4.819 | 3341.811 +/- 5.600 | 95.515% | 149.871 |

Graphs speed healthy by12.08% and NEP by11.81%. Absolute NEP overhead falls
9.17ms, but owner parity does not improve because healthy speeds up slightly more.

### Ordinary Backward And Host Submission

Means per ordinary microbatch (indices1-6), averaging all ranks of the second
replica. GPU durations are schedule-level elapsed intervals, not pure GEMM time.
CPU values are summed thread-CPU time in native expert backward/recompute scopes.

| Metric | Router healthy | Router reduced NEP | Full-MoE healthy | Full-MoE reduced NEP |
| --- | ---: | ---: | ---: | ---: |
| Ordinary backward elapsed ms | 291.632 | 295.480 | 233.525 | 235.537 |
| Eager expert backward/recompute CPU ms | 85.105 | 104.401 | 0 | 0 |
| Full-MoE backward replay CPU ms | N/A | N/A | 1.339 | 1.410 |

Full-MoE profiles have120 MoE backward replays per iteration (15 layers x GA8)
and no eager expert-forward/backward calls on any rank. The reduced replica's
ordinary-backward gap decreases from3.848ms to2.011ms. Graphs remove the native
per-expert Python submission path, but most remaining end-to-end NEP overhead
in this static-routing workload lies elsewhere.

### Owner-Time Difference By Nonoverlapping Schedule Phase

These phase differences sum to the measured iteration difference. They are
not overlapping collective residencies and do not assign a transfer-only cost.
Outside F/B includes gradient finalization, optimizer, and schedule gaps.

| NEP minus healthy, ms per iteration | Router only | Full MoE |
| --- | ---: | ---: |
| First forward | 87.315 | 94.273 |
| All other forwards | 3.480 | 0.997 |
| All non-final backwards | 10.190 | -5.333 |
| Final backward | 38.661 | 39.862 |
| Outside F/B ranges | 19.398 | 20.072 |
| Total | 159.045 | 149.871 |

### Instrumentation And Numerical Limits

Post-profile logged parity (steps36-40) is95.792% router-only and95.512% full
MoE, within0.011 percentage points of the matched measured parity. Pre-profile
15-23 parity is96.070%/95.731%; warmup/drift can contribute. These controls are
not substituted for measured iterations and are not a causal profiler-cost estimate.

Padding validation smoke3120612 completed all12 steps in dropless, padded-router,
and full-MoE modes on EP8/6. Dropless versus padded-router logged loss/gradnorm
values match exactly at every step; measured8-10 times71.0 versus71.1ms.
This directly validates the padding-only path at the logged precision.

Full-graph numerical equivalence is NOT established: all four large runs are
finite with no skipped/NaN iterations, and the first two logged loss/gradnorm
values match within each topology. Small differences begin with replay and
grow later in this high-learning-rate dummy-data run, in healthy as well as NEP.
Maximum relative logged loss differences reach1162% healthy/656% NEP. This is
not proof of harmless rounding or a NEP-specific error. A controlled one-step
output/gradient comparison is needed before accepting full-graph training
equivalence; the results above are performance measurements on fixed routing.

Artifacts:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3120733`.
Full rank-role statistics and per-iteration samples are in
`native_moe_graph_analysis.json`. Analysis script:
`scripts/nonuniform/analyze_a8b_native_moe_graph.py`.
