# Earlier next-bucket AG versus current-bucket redistribution

## Implementation and validation

- Isolated branch `nep-ag-red-overlap-20260922`, commit `ccb4b9726`, based on
  accepted cleanup `ee28f09ff`. Trusted branches are unchanged.
- Only production change: dispatch the next native parameter AG before inserting
  the current bucket's parameter-ready wait on the compute stream in
  `NonuniformEPNCCLParamAndGradBucketGroup.finish_param_sync`.
- No new streams, buffers, collectives, or parameter layouts. Native AGs remain
  ordered on their existing stream. Redistribution remains serialized on its shared
  stream, waits for its own AG, and gates its consuming compute.
- Job3131771:83 unit tests passed, including four new ordering/lifecycle cases.
  Four-GPU smoke verified fresh parameters, followers, repeated buffer reuse,
  force-sync, native model-EP/AG/RS progress, and next-AG GPU completion before
  deliberately delayed prior redistribution. No imports changed in committed code.
- Job3131835: all three35-iteration cases completed without hangs, skipped steps,
  or NaNs. All88 rank trace sets contain native AG/RS and NEP records as applicable.
  This is targeted correctness evidence, not a bitwise training-equivalence claim.

## Matched experiment

GB300,8nodes segment4, replicas wholly within distinct NVLink domains18/20.
A8B latent,15MoE layers in the retained hybrid/MTP pattern, H4608, latent1152,
expert F3072,512experts,topk6,sequence8192,MBS1,GA2,TP2,CP1,ETP1,PP1.
HealthyEP16/16,GBS32,localBS16/16; NEPEP16/12,GBS28,localBS16/12.
Flex/HybridEP,DistOpt,native MoE graphs,uniform padded routing,backward SM margin32.
Same Nsight cuda-sw graph-node capture and CUDA-event metadata on every rank.
Measured steps20-27 after19warmups; owner ranks0-11. Healthy retains5expert buckets
and NEP8; this inherited healthy/NEP difference is unchanged in the before/after A/B.
All candidate/control workload options match apart from output paths and code SHA.

| Owner mean, ms | Healthy | Accepted NEP | Earlier AG |
|---|---:|---:|---:|
| First forward |94.792|133.644|131.957|
| Other forward |79.223|79.766|79.653|
| Nonfinal backward |120.349|120.661|120.432|
| Final backward |134.375|156.332|156.955|
| Outside forward/backward |58.127|72.299|71.522|
| Iteration |486.866|562.702|560.519|
| First-forward parity |100%|70.93%|71.84%|
| Iteration parity |100%|86.52%|86.86%|

First-forward gain1.686ms; iteration gain2.183ms. Iteration SD1.128/1.301/1.637ms;
first-forward SD0.299/0.577/0.675ms. Final-backward increase0.623ms is smaller than
the per-case SD1.456/1.359ms; this run does not establish a backward regression or
a guarantee of no regression on other workloads. Maximum allocated memory is
identical between NEP variants:116.394GiB.

Profiling remains enabled. Logged measured-minus-post-window differences are
13.423ms healthy,12.945ms accepted NEP,13.632ms candidate. These are distortion
diagnostics, not corrections; all table values use matched measured iterations.

## What the kernels show

| First-forward owner mean, ms | Accepted NEP | Earlier AG |
|---|---:|---:|
| Native expert AG kernel union |55.760|55.924|
| Redistribution kernel union, including staging |29.773|39.166|
| Intersection of those kernel intervals |0.000|13.546|
| Broadcast kernel residency |23.681|33.091|
| Broadcast before last peer kernel starts |21.236|30.673|
| Broadcast after last peer kernel starts |2.445|2.418|
| Native model-EP-only time inside graphs |27.479|25.091|

AG and redistribution actually overlap. Native AG kernels remain serialized;
redistribution kernels remain serialized and begin after their own AG kernels.
The extra9.410ms of broadcast residency is almost entirely before the last peer's
broadcast kernel starts, not increased post-last-peer duration. Kernel-start prefixes
are peer-arrival diagnostics, not pure transport service or additive recoverable costs.

Rank11 is still the last full-replica owner ready for all64bucket/step pairs.
The last bucket becomes ready101.522 ->99.845ms after rank0's first-forward start:
only1.677ms earlier, matching the1.686ms first-forward improvement. The mean
full-replica parameter-ready spread remains6.159 ->6.379ms. Earlier AGs do not
eliminate the ordered follower-delivery chain; much of the new concurrency is
early owners waiting for peers, rather than all-rank parameter availability advancing.

The prior26ms estimate was a rank-local launch-opportunity calculation, not a
critical-path speedup bound. This experiment does not recover that amount and does
not establish NVLink bandwidth contention as the explanation. Retain this as an
isolated, modest improvement; no further runtime changes or experiments were made.

## Follow-up: the cross-follower delivery chain

In step20's third parameter bucket, relative to rank0 first-forward start, owner0
AG completes26.452 ->24.916ms (1.536ms earlier); owner11 AG completes32.213 ->24.557ms
(7.656ms earlier). Owner11 parameters become ready36.790 ->35.035ms: only1.755ms earlier.

Matched memberships show follower12 receives owners0/1/2, follower13 receives2/3/4/5,
follower14 receives5/6/7/8, and follower15 receives8/9/10/11. Transfers for owners2,5,8
share adjacent followers, linking their serialized streams into a delivery chain.
In the candidate, follower15 starts the owner8 broadcast24.440ms but follower14
reaches it only30.889ms. Owner11 starts its broadcast24.945ms; follower15 reaches it
33.284ms. Earlier late-owner AG completion therefore becomes additional waiting.

The26ms estimate used the late owner's local launch opportunity without modeling
this cross-rank chain. The same old trace had only2.296ms opportunity on owner0.
It should not have been presented as a useful first-forward speedup estimate without
establishing how much the entire delivery chain could advance. No new jobs or code
changes were needed for this follow-up.

## Reproduction and analysis caveats

Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3131835/`.
Phase table: `cleanup_phase_analysis.json`.
Actual overlap and peer matching: `ag_redistribution_analysis/overlap_results.json`.
Last-owner ready times: `ag_redistribution_analysis/parameter_readiness.json`.

Run `python3 scripts/nonuniform/analyze_nep_cleanup_ab.py 3131835 --hard`, then
`python3 scripts/nonuniform/analyze_nep_ag_red_overlap.py 3131835`.
The newer offline parser aligns CPU clocks per iteration and assigns captured
graph kernels using their launch correlation/NVTX ranges, not approximate GPU
window intersection. This avoids small clock drift misassigning boundary kernels.
CUDA-event and Nsight GPU timestamps can differ by roughly0.1ms; peer timing is
therefore reported to0.1ms when interpreting mechanisms, not as exact wire timing.
Kernel concurrency/order tests compare Nsight kernels in the same rank/clock.
Early parser attempts are superseded by schema4 caches and final overlap JSON.

The inherited generic runner emits a forced-load-balancing validation complaint
because this workload uses exact-uniform routing instead. The dedicated hard-run
validator checks that exact-uniform flag, all35finite iterations, batch settings,
and complete trace sets; it passed for all cases. No training failure is hidden.
