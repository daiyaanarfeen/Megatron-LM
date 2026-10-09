# Strong First-Forward / Final-Backward Workload Sensitivity

## Result

Job3063815 completed0:0 in29m13s on2026-09-15, GB200lyris[0014-0017].
The two-knob perturbation lowers measured owner parity from95.60-95.71% to
80.73%. Its positive temporal penalties are concentrated in first forward,
final backward, and the following optimizer boundary, not ordinary microbatches.
The substantial degradation also exists before and after recording. This is
evidence of workload-dependent exposure/amortization, not an implementation edit.

| Case | Healthy mean +/- SD ms | NEP mean +/- SD ms | Owner parity | Pre / post parity |
| --- | ---: | ---: | ---: | ---: |
| Anchor start |705.42 +/-4.50|737.93 +/-6.66|95.60%|95.24 /95.52%|
| Expert F768 ->3072 |776.22 +/-1.72|865.02 +/-5.36|89.73%|89.83 /90.19%|
| F3072 + sequence8192 ->2048 |370.09 +/-2.17|458.45 +/-6.95|80.73%|79.55 /78.29%|
| Anchor end |700.02 +/-9.29|731.41 +/-8.63|95.71%|94.04 /94.63%|

Each mean uses the same recorded iterations40-49. Average full-owner ranks0-5
within iteration, then average ten iterations; SD is across these iteration
means. Pre29-38 and post51-60 are separate logged diagnostics, never substituted
for the recorded timing or used for subtractive profiler correction.

## Phase Evidence

NEP full-owner minus healthy owner temporal spans, milliseconds. LaterF and
earlierB each SUM two ordinary microbatches; firstF and finalB each occur once.
The intervals use first/last grouped-expert GPU operations as boundaries, not
whole-model phase endpoints. They include intervening layers and dependencies.

| Case | FirstF | LaterF total | EarlierB total | FinalB | LastB -> optimizer | Other intervals | Total iteration gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Anchor start |16.22|7.49|-0.21|2.74|2.40|3.86|32.51|
| F3072 |53.40|11.15|0.17|9.67|3.94|10.48|88.80|
| F3072 + seq2048 |44.16|-4.81|-5.98|24.47|22.78|7.74|88.36|
| Anchor end |17.70|10.78|0.01|-0.97|-3.16|7.03|31.39|

The full unrounded additive partition is in sync_workload_analysis/summary.json.
These are disjoint elapsed intervals, not overlapping sums of NCCL residency.
Negative entries are retained; positive entries alone can exceed the total gap.

### Width Increases Parameter Delivery Cost

The number of parameter-redistribution calls remains8. Their summed GPU elapsed
envelopes increase9.83 ->33.95ms when expert F quadruples, while summed CPU wall
time stays9.25 ->8.90ms. First-forward excess relative to ordinary forward,
after the same healthy subtraction, increases12.48 ->47.82ms. Earlier backward
changes only0.09ms per microbatch relative to healthy. Thus the data do not
support more parameter-staging CPU work as the explanation for this width effect.

GPU operation envelopes include dependencies and are NOT pure transfer times.
They cannot resolve exact NCCL bandwidth, GPU starvation, or isolate each
collective's causal contribution. Native bucketing also changes with width.

### Replica Progress Is Not The Owner Temporal Partition

Ordinary microbatches do not synchronize gradients across replicas. Slower
ordinary work on the reduced replica can therefore appear as extra waiting in
the owner's FINAL backward, without lengthening the owner's earlier backward.
The owner partition above localizes elapsed time; it cannot rule this cause out.

For F3072+seq2048, NEP full/reduced laterF averages49.06/55.55ms and earlierB
38.94/44.24ms. Reduced firstF is shorter92.22 versus97.59ms, but its last
expert-backward completion occurs418.06ms after its local iteration start,
versus402.99ms on full owners. AtF3072/seq8192 the reduced cohort instead
finishes at804.37ms versus824.70ms on full owners. These local-start-relative
measurements suggest that ordinary reduced work can consume its initial lead
in the shorter-sequence case. They are NOT an exact cross-rank rendezvous delay
without validating iteration-start alignment and all participants' readiness.
Do not describe the80.73% case as proving that all slowdown is transfer service
time or that reduced-replica ordinary work contributes nothing.

### Shorter Sequences Reduce Available Computation

At fixedF3072 and fixed parameter bytes/buckets, sequence8192 ->2048 lowers
healthy iteration776.22 ->370.09ms and NEP865.02 ->458.45ms. The absolute gap
is almost unchanged88.80 ->88.36ms, so its relative impact roughly doubles.
This comparison holds GA3 fixed: it is NOT gradient-accumulation amortization.

Parameter-delivery GPU envelopes remain33.95 ->34.15ms and CPU8.90 ->8.99ms.
Gradient-staging CPU wall time remains10.39 ->10.55ms, but mean GPU-start lag
behind CPU invocation falls79.41 ->14.56ms. FirstF excess over ordinaryF stays
47.82 ->46.57ms; finalB excess over ordinaryB increases9.58 ->27.46ms.
LastB-to-optimizer NEP-minus-healthy gap grows3.94 ->22.78ms. This localizes the
added relative burden to sync-adjacent phases. It does NOT establish exactly
which uninstrumented NCCL operation occupies every millisecond of the tail.

## Controls and Limits

- Accepted runtime93d313df4 unchanged; nonuniform_ep.py SHA256
  653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7.
- Anchor:15Mamba +15MoE +3GQA blocks,H1280,F768,sharedF1536,E128,topk6,
  MBS2,GA3,sequence8192,TP/CP/ETP/PP1,FlexHybridEP,DistOpt and existing graphs.
- HealthyEP8/8 versusNEPEP8/6, proportional replica batches48/48 versus48/36,
  GBS96/84. Only expertF, then sequence, changed as independent settings.
- Four nodes,segment4,replicas contained in nvlblk01. Software cliques1000/1001
  enforce cross-replicaRDMA: each tag has30NCCL logs,28withtwo-rankcliqueSize1,
  and1344RDMA channel records. Same allocation/container for all eight cases.
- RequestedDDP8 unchanged, but automatic physical buckets change with width:
  anchor3dense/4expert;F3072 has2dense/6expert. NEP uses8logicalowner groups
  throughout. Thus width includes derived native rebucketing and is NOT a
  pure payload-only or iso-bucket comparison. The sequence contrast atF3072
  does preserve bucket signatures and payload.
- All8 cases complete60steps with finite losses,no skips/noNaNs; all120rank
  profiles/1200rank-iterations pass schema,clock,event-count checks. No kernel
  profiler added or disabled; same low-overhead targetedCPU/CUDAevent protocol.
- One allocation,not independent repeated allocations. Measured anchor parity
  stable,but final anchor and short-sequence surrounding windows still drift.
  This does not invalidate the large below85% drop in every short-sequence
  window; it prevents treating80.73% as an invariant exact value.
- Job requested bothGB200/GB300 backfill; allocatedGB200. No claim that these
  measurements compare hardware generations.

## Artifacts

Lustre root:
/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3063815

Summary and validation:sync_workload_analysis/{summary,final_validation}.json.
Raw events:TAG/cupti/ep8/scaling_ladder__a770m_4b_moe_215b/CASE/3063815/cupti_profile.

Launcher:scripts/nonuniform/run_nep_sync_stress_20260915.sh.
Argument-only entrypoint:scripts/nonuniform/pretrain_nep_sync_stress.py.
Reproduce analysis:

```bash
python3 scripts/nonuniform/analyze_nep_sync_workload.py 3063815 --tags anchor_start expert_f3072 expert_f3072_seq2048 anchor_end
```
