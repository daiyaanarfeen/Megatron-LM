# a8b latent MBS2 overhead audit

## Result and scope

Job3078207: original a8b_120b_latentmoe_1t, H4608, latent1152,
expert F3072,512 experts,top-k6,30 MoE layers including MTP. MBS2,GA8,
sequence8192,TP2,CP1,ETP1,DistOpt,Flex/HybridEP. HealthyEP16/16,NEPEP16/12;
GBS256/224,replica batches128/128 versus128/96. Proportional allocation.

| Window | Healthy ms | NEP ms | Owner parity | NEP minus healthy ms |
| --- | ---: | ---: | ---: | ---: |
| Profiled iterations25-34 | 8444.05 | 9181.22 | 91.971% | 737.17 |
| Post-profile iterations36-45 | 8358.31 | 9082.03 | 92.031% | 723.72 |

Both MBS2 cases completed45iterations,zero skipped/NaN steps,all32/28rank
profiles. Post-profile throughput7840.8healthy/7216.0NEP tokens/s/GPU.
Replicas each fit one physical domain(nvlblk21/nvlblk26); cross-replica
NET/NCCL RDMA/GDRDMA confirmed. Same named NeMo26.06 image and pinned runtime
hashes as3074709. No training-code changes for this experiment or audit.

The allocation was preempted after30m28s,after MBS1 healthy completed and
before MBS1 NEP started. This does not invalidate the complete MBS2 pair.
User clarified same-allocation MBS1 controls are unnecessary; control the
hardware/resources/topology instead. No replacement allocation submitted.

## Non-overlapping elapsed-time accounting

Use ONLY profiled iterations25-34 to explain their737.17ms gap, not the
post-profile723.72ms gap. Mean across full-replica owner ranks0-11,then
across the ten iterations. Both expert forward and backward events are
parsed into eight microbatches; selective recomputation belongs to backward.

| Interval | Healthy ms | NEP ms | Extra ms |
| --- | ---: | ---: | ---: |
| Iteration start through first forward's last expert | 287.682 | 673.802 | 386.119 |
| Other seven forward expert spans | 1716.668 | 1800.491 | 83.823 |
| First seven backward spans,including expert recomputation | 4743.765 | 4858.354 | 114.588 |
| Final backward span,including expert recomputation | 698.347 | 810.286 | 111.938 |
| Last expert backward end to optimizer start | 76.622 | 91.947 | 15.325 |
| Optimizer step | 35.367 | 51.547 | 16.180 |
| After optimizer to iteration end | 4.988 | 5.570 | 0.582 |
| Forward/backward and inter-microbatch transitions | 880.914 | 889.537 | 8.624 |
| Total | 8444.353 | 9181.534 | 737.180 |

The0.01ms difference from logged delta is iteration-marker/logging overhead.
Every rank/iteration partition sums exactly before display rounding. These
are elapsed intervals bounded by expert operation events, NOT complete
model-forward/backward kernel scopes and NOT pure compute time. Transitions
include unrecorded attention/Mamba,loss,MTP and scheduling work.

## Largest source: first-forward parameter delivery

This is not the previously fixed late CPU submission. The current
start_param_sync queues redistribution immediately behind AG on the transfer
stream. In iteration25,the third bucket's redistribution CPU calls are
submitted around109-113ms,BEFORE AG completes around123ms. The later delay
is on the GPU dependency/communication path.

The full replica uses these owner-to-holder parameter-broadcast groups:

```
(0,12) (1,12) (2,12,13) (3,13) (4,13) (5,13,14)
(6,14) (7,14) (8,14,15) (9,15) (10,15) (11,15)
```

Each rank processes its transfers in owner order on one shared transfer
stream. Groups spanning adjacent followers connect their stream dependencies:
owner2 links followers12and13,owner5 links13and14,owner8 links14and15.
This creates a serialized delivery chain across all12owners,even though
all owners' AGs have already completed. The trace shows the resulting order.

Iteration25,third bucket,ms relative to rank0 iteration start:

| Owner rank | AG complete | Redistribution complete |
| --- | ---: | ---: |
| 0 | 122.9 | 130.1 |
| 3 | 123.0 | 140.5 |
| 6 | 122.8 | 151.9 |
| 9 | 122.3 | 163.1 |
| 11 | 122.7 | 170.6 |

The following expert layer begins across the full replica at171.9-172.0ms.
An early owner finishing redistribution at130ms cannot advance its native
EP-dependent expert computation until the late holders are ready. We do not
have individual HybridEP kernels here,but the ordering constraint is in the
code and the resulting cross-rank expert start alignment is in every trace.

Across all ten iterations,the final six buckets each take47.8-49.1ms from
the latest full-owner AG completion to the latest full-replica redistribution
completion. Transfer completion spreads across ranks are approximately40.8ms.
Before forward MoE ordinals4,7,11,15,19,23(zero-based),the first microbatch
has approximately50.6ms more gap than the next microbatch. The same stalls
repeat at parameter bucket boundaries,not throughout all expert execution.

### Excess parameter payload

The parameter path packs and broadcasts the ENTIRE fixed-width43-expert owner
row,then a follower copies only its locally held subset. Each follower holds
32true experts,but participates in3or4owner broadcast groups per bucket.

Measured totals per full-replica follower per iteration:

- Buffers received through parameter broadcast:68.4785664GB.
- Useful routed-expert parameters:13.58954496GB(the native healthy expert AG
  output for32experts/GPU,and half that follower's27.17908992GB FP32 gradient
  Gather send payload).
- Ratio:5.0390625x. This is logical receive-buffer volume,NOT a hardware
  NVLink byte-counter measurement. There is no claim of exactly5x speedup
  from removing it.

Thus NVLink transfers may be individually fast while many unnecessarily
large dependent transfers produce a large critical-path delay.

### Wait is not wire time

Full-owner parameter broadcasts total171.140ms/rank/iteration. Matching
all participants' input-ready events assigns147.631ms before the latest
participant becomes ready,and23.510ms after that point. Followers total
309.783ms,of which208.245ms is before latest readiness and101.538ms after.
These sums are not additive iteration overheads. Post-readiness ranges still
include queueing/protocol/completion effects; they are NOT isolated wire time.

The reduced replica has no remote parameter redistribution,just local copies.
Its AG residency is529.631ms,but419.618ms is before the full-owner partner
is ready. That is a consequence of the delayed full replica,not evidence
that reduced-EP AG bandwidth is poor.

Relevant current code: nonuniform_ep.py _get_nep_distopt_param_transfer_plan
(1628),_scatter_nep_distopt_owner_params_to_physical_holders(1699),
start_param_sync(1749),finish_param_sync(1796). The latter enforces the
necessary consume-after-delivery dependency. This audit does not recommend
removing correctness dependencies.

## Final backward and optimizer

Final backward adds111.938ms on full owners. Its grouped expert backward+
recompute ranges actually decrease slightly:137.754ms healthy versus135.600ms
NEP. CPU time inside those grouped expert calls is also almost unchanged.
The added elapsed time is outside those expert scopes,where gradient packing,
Gather,owner-layout staging,RS,and remaining native work execute.

Measured full-owner totals(overlap means these must NOT be added):

| Operation | Healthy | NEP |
| --- | ---: | ---: |
| Expert RS calls per iteration | 6 | 8 |
| Expert RS input GB | 27.179 | 36.239 |
| Expert RS summed elapsed ms | 192.766 | 231.836 |
| Expert RS union within final backward ms | 130.78 | 203.68 |
| NEP gradient Gather elapsed ms | absent | 30.230 |
| Owner gradient staging GPU elapsed ms | absent | 22.615 |
| Owner gradient staging CPU wall ms | absent | 43.281 |

The33.3%larger owner gradient payload is expected with16/12EP. Both runs
request8DDP buckets,but native healthy actually forms6expert buckets while
NEP forms8. Dense gradient buckets remain3. They are not iso-expert-bucket
runs; this is the original script's native bucketing versus current NEP.

The reduced replica enters final backward73.1ms BEFORE its full-owner peer
and finishes its last expert backward only2.3ms later. The LAST RS waits
only1.83ms for the reduced peer on average. After both sides are ready,last
RS takes26.33ms NEP versus31.79ms healthy. A large late-reduced-peer wait is
NOT the explanation for this run's tail. Full-owner readiness after its last
expert grows0.87->9.55ms; total backward-to-optimizer grows15.32ms.

Optimizer itself grows16.18ms,consistent with owners updating more expert
state. This is the whole optimizer scope,not an isolated optimizer-kernel
measurement,so the exact extra16.18ms cannot all be assigned to arithmetic.

## Ordinary microbatches: localized but not fully attributed

Full-owner ordinary forward averages245.238->257.213ms;ordinary backward
677.681->694.051ms. Across7of each,this adds198.411ms. No NEP parameter
redistribution or gradient Gather/RS runs in these ordinary intervals.
Consequently this portion cannot simply be called exposed NEP reshards.

Full-owner ordinary backward+recompute expert ranges are138.563ms healthy
versus134.689ms NEP;their CPU scopes are165.044versus164.530ms. Larger/slower
full-owner expert GEMMs do NOT explain the additional backward span.

Reduced-rank ordinary backward+recompute expert ranges do increase:
131.017->173.212ms;CPU scopes157.665->196.570ms. This is consistent with
42/43experts per reduced GPU versus32on healthy. These are operation ranges,
not active GEMM kernel times,and do not establish a GEMM efficiency ratio.
Moreover,the reduced replica remains ahead entering final backward,so its
larger expert scope is not itself evidence of an iteration bottleneck.

The profiles do not instrument individual HybridEP dispatch/combine kernels,
Mamba/attention graph replay,or all DDP/autograd hooks. Exact attribution of
the198.411ms ordinary-microbatch difference among those operations requires
additional targeted scopes. It is currently UNRESOLVED,not automatically
CPU starvation,network contention,or reduced GEMM efficiency.

## MBS1 comparison and conclusion

Existing resource-matched GB300 MBS1 job3074709 has382.382ms additional
startup/first-forward time,almost identical to MBS2's386.119ms. Increasing
MBS has not removed the dominant parameter-delivery mechanism.

The complete observed post-profile gap shrinks only746.40->723.72ms while
healthy iteration time grows5707.58->8358.31ms. Improved parity therefore
does not imply this fixed overhead was eliminated.

Highest-value implementation target: send only each holder's required
parameters and remove avoidable serialization between otherwise independent
owner-to-holder deliveries,while preserving AG->redistribution->parameter-use
ordering and transfer-buffer lifetime. No implementation change made here.

A smaller first experiment can retain the existing process groups and ordering:
pack only the experts offloaded to followers, and copy owner-local parameters
locally. For this placement, current follower receives total 645 expert-row
equivalents (15 receiver memberships times width 43). Offloaded-only broadcasts
would send 160: 128 useful experts plus 32 duplicated across bridging groups.
That reduces average follower receive payload from 68.479 GB to 16.987 GB,
about fourfold, without initially changing collective ordering. This is a
payload estimate, not a predicted iteration speedup; packing cost and preserved
serialization still require measurement. No such change has been implemented.

## Evidence and limitations

- Analysis covers all60ranks and ten matched iterations per case,with exact
  microbatch event-count,participant-set,and additive-partition assertions.
- CPU/CUDA timestamp alignment drifts by up to about1.5ms in some records;
  do not use sub-ms host/GPU comparisons as proof of starvation. Cross-peer
  AG/RS completion discrepancies are about1ms or less. These uncertainties
  are small versus the repeated40-50msdelivery-chain stalls.
- GPU ranges include internal launch/dependency gaps. No exact kernel busy,
  exposed NCCL,or physical link-bandwidth claim is made from them.
- Analysis artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3078207/mbs2_overhead_analysis/{summary.json,delivery_chain.json}.
- Reproduce: scripts/nonuniform/analyze_a8b_current_overheads.py and
  scripts/nonuniform/analyze_a8b_delivery_chain.py. MBS1 comparison uses the
  former with `--job 3074709 --tag current`.
