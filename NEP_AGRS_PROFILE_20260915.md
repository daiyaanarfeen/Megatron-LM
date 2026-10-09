# Native AG/RS and NEP reshard completion profile

## Completed experiment

- Job3065919 COMPLETED0:0; GB200 lyris[0039,0047,0049-0050], segment4.
- Healthy then NEP on the same four-node allocation. Software NVLink
  cliques1000/1001 keep each replica NVLink-local. NCCL logs confirm
  cross-replica connections use NET/NCCL RDMA Plugin v11/GDRDMA, not NVLink.
- a770-family hybrid:15 MoE,15 Mamba,3 GQA layers; H1280, expert F768,
  shared F1536,512 experts,top-k6,sequence2048,MBS2,GA3,TP/CP/PP/ETP1.
- Healthy EP8/8, replica BS48/48,GBS96; NEP EP8/6, replica BS48/36,GBS84.
  DistOpt,Flex/HybridEP, original balanced-random routing and local graph
  settings retained. Runtime93d313df4d55 unchanged. This was selected because
  prior job3064000 had a strong final-backward penalty (~68.6ms).
- Original bucketing retained:2 native dense buckets,7 healthy expert buckets,
  8 NEP logical expert groups. This is a workload replay, not an iso-bucket test.
- Both cases completed60 iterations, no skipped/NaN iterations. All16 healthy
  and14 NEP rank profiles cover40-49. Analysis validates required AG/RS records,
  collective participants/order and matching AG/RS payloads across every PG.

## Iteration and phase timing

Means over iterations40-49 and full-replica owner ranks0-5. Phase boundaries
are first/last recorded grouped-expert GPU operations, NOT complete layer
compute kernels. SD below is across10 iteration means, not independent ranks.

| Measurement (ms) | Healthy | NEP | NEP minus healthy |
|---|---:|---:|---:|
| Owner iteration |671.969 +/-7.442|894.533 +/-7.837|222.564|
| First-forward expert envelope |90.010|144.016|54.006|
| Later-forward envelope, per microbatch |82.892|81.776|-1.116|
| Earlier-backward envelope, per microbatch |93.442|95.721|2.280|
| Final-backward expert envelope |98.978|178.532|79.553|
| Last expert backward to optimizer start |19.956|96.373|76.417|

Owner parity is75.12%, or33.12% longer iteration time. This is the new measured
result, not the prior81.64% result. The allocation changed; absolute healthy
time improved more than NEP versus the old job. Do not attribute that change
solely to profiling or claim reproduction of the old parity.

Rank0 logged-window diagnostics (not corrections):

| Window | Healthy ms | NEP ms | Parity |
|---|---:|---:|---:|
| Pre-profile29-38 |664.93|897.11|74.12%|
| Profiled40-49 |671.72|894.27|75.11%|
| Post-profile51-60 |661.34|874.11|75.66%|

The recorded-minus-post differences are10.38ms healthy and20.16ms NEP.
The differential is9.78ms, but NEP recorded timing is actually2.84ms below
its pre-window. Runtime drift and instrumentation are not separable from
these windows alone. No timing is corrected or replaced using another window.
Mean measured observer-enqueue CPU time is1.39ms healthy versus2.21ms NEP
per full owner/iteration; this excludes other profiling overhead and is not
equivalent to exposed GPU delay.

## Collective completion measurements

Numbers are per-rank interval unions, then averaged over owner ranks0-5 and
the same10 iterations. A range starts when its issuing stream reaches the
input-ready event and ends on an observer stream after the existing native
Work completes. It includes host submission gaps, queueing and peer waiting.
It is NOT kernel-only NCCL residency, transfer-only time, or exposed overhead.

| Outstanding operation (ms/iteration) | Healthy full | NEP full | NEP reduced |
|---|---:|---:|---:|
| Expert parameter AllGather |22.707|30.619|113.512|
| Expert gradient ReduceScatter |45.196|239.128|61.793|
| NEP intra-replica gradient Gather |0|141.179|0|
| NEP parameter broadcasts |0|28.164|0|
| Complete NEP parameter redistribution stage |0|44.136|5.162|
| NEP gradient staging |0|17.259|30.142|

The last two rows are parent operation ranges, not additional collectives.
Broadcasts are inside parameter redistribution. Different rows overlap and
must NOT be summed to explain the222.564ms iteration gap. Reduced EP6
physical holders already match owner placement, so there is no remote Gather
or parameter broadcast there; local gradient/parameter staging still exists.

Healthy versus NEP full-owner AG output is3.775 versus5.033GB/iteration;
RS input is7.550 versus10.066GB/iteration: exactly4/3 on average, as expected
for8 versus6 owner lanes. These are tensor payload bytes, not wire traffic.

For full owners,37.254ms healthy versus164.012ms NEP RS-pending union occurs
inside the final-backward expert envelope. After the final expert operation
and before optimizer start, RS is still pending for7.942 versus75.116ms;
NEP Gather is pending for36.597ms in that same tail. Those tail windows also
contain other backward operations/dense sync and are not proven fully exposed.

### What the data establishes

- Native AG/RS completion is now measured separately from NEP resharding.
- RS is the longest outstanding final-backward collective on full owners.
  It is NOT valid to conclude that239ms was spent transferring RS bytes:
  reduced owners report61.793ms, and their peers' queue/arrival differences
  contribute to the full-owner ranges. Healthy has the reverse asymmetry:
  second-replica owners have120.893ms RS-pending union versus45.196ms on full.
- NEP first-forward parameter redistribution itself lasts44.136ms, including
  28.164ms broadcast ranges plus staging/launch/dependency spacing. It is
  independent of the30.619ms native AG measurement, but their overlap must be
  considered before attributing first-forward latency.
- Additional attribution requires aligning per-bucket readiness/completion;
  these totals alone do not identify a unique cause of final-backward delay.
  Attention/Mamba/HybridEP kernels are not individually captured, so time
  outside recorded expert ranges cannot be labelled GPU idle or exposed comm.

## Follow-up: proportions and post-AG prefetch opportunity

This run is potentially differentially distorted by profiling. Earlier work
already established NEP's greater sensitivity to profiling; pre/post windows
do not rule out the new instrumentation because its wrappers stay installed
and perform work even while recording is inactive. No allocation-causal or
unbiased-parity conclusion is supported by this run alone.

Within the recorded first-forward envelope, AG-pending union is16.022ms
healthy versus25.760ms NEP, an increase of9.738ms. That magnitude is18.0%
of the54.006ms forward-envelope increase. NEP redistribution occupies41.912ms
inside that envelope,77.6% of the increase (the initial2.223ms occurs before
the first expert marker). These are magnitude comparisons, NOT causal
percentages: AG includes waiting and overlaps other work. AG/redistribution
mutual overlap is only0.004ms/iteration in this capture.

Within final backward, RS-pending union increases37.254->164.012ms, or126.757ms,
which exceeds the79.553ms envelope increase. Therefore RS pending duration
cannot be assigned as an additive fraction of backward overhead. After the
last expert finishes, all local Gather/staging completes38.521ms later;
the final expert RS completes75.116ms later, leaving36.595ms of RS pending
after local Gather/staging has entirely finished. This can still include
remote participant waiting, not just transfer service.

Matched per-bucket GPU events, full owners0-5, all10 measured iterations:

| Parameter bucket | AG duration | AG done to redistribution start | Redistribution duration | Recorded expert-range coverage after AG |
|---|---:|---:|---:|---:|
| Initial0 |2.518ms|approximately0|2.223ms|0|
| Next1 |4.016ms|2.979ms|6.094ms|0.921ms|
| Middle2-7, range of bucket means |4.002-4.024ms|9.612-10.921ms|5.908-6.001ms|3.471-4.318ms|

For all360 middle-bucket samples, AG completes before the previous expert's
last forward operation finishes. Redistribution remains submitted late:
CPU submission follows previous expert completion by about1.9-2.0ms.
There is no recorded redistribution/expert GPU-range overlap.

The middle-bucket means conceal owner asymmetry. Owner0 has12.679ms between
AG completion and redistribution versus3.460ms redistribution; owner5 has
7.920ms versus8.223ms. Thus there is opportunity, not a guarantee all owners
can hide redistribution completely. The available elapsed interval includes
non-expert work, host gaps and model-EP; it is NOT9-11ms of pure compute.
Only3.5-4.3ms is covered by recorded grouped-expert ranges (also not pure
kernel service), and attention/Mamba kernels remain individually unmeasured.

Concrete owner0/iteration44 example, relative to first expert forward start:
bucket2 AG finishes13.065ms; preceding expert layers1/2 have operation spans
14.746-18.057ms and20.608-23.750ms; redistribution is delayed until25.774ms
and finishes29.367ms. This demonstrates AG completion before useful prior
layer work, followed by late redistribution, without claiming continuous
compute or contention-free overlap over the entire interval.

## Reproduction and artifacts

### Final RS: matched peer-readiness explanation

Matched the final RS on each full owner0-5 to its EP6 peer8-13 across
iterations40-49. Timestamps use each trace's origin_ns plus calibrated GPU
event time. Paired final completion timestamps differ by-0.097 to+0.021ms,
small relative to the32ms readiness gap (not an independent clock calibration).

Relative to each full owner's last local Gather/staging completion:

| Milestone | Mean time |
|---|---:|
| Full-owner final RS input-ready |+0.002ms|
| Reduced peer final recorded expert backward completes |+28.762ms|
| Reduced peer final staging starts |+30.442ms|
| Reduced peer final staging completes |+32.100ms|
| Reduced peer final RS input-ready |+32.165ms|
| Full-owner final RS completion |+36.595ms|

Thus32.165ms of the36.595ms tail precedes peer input readiness; only4.430ms
remains afterward. The latter still includes scheduling/transport and is not
pure wire time. Most of the readiness delay precedes the reduced peer's final
expert-backward completion, not its final staging copy (mean1.657ms).

Example iteration44, owner0/peer8: owner0 final RS ready+0.003ms; rank8's
last expert backward ends+36.573ms, staging ends+37.954ms, RS ready+37.956ms;
owner0 RS completes+42.095ms. Rank8's CPU RS submission was already at
+20.083ms, so its GPU operation was waiting for preceding work/dependencies,
not for a host launch postponed until+37.956ms.

This identifies the final RS peer-readiness dependency. It does not yet split
the reduced peer's backward delay into GEMM execution, host gaps, native
model-EP, or profiler distortion. No new jobs or runtime changes.

- Launch: `sbatch scripts/nonuniform/run_nep_agrs_profile_20260915.sh`.
- Analyze: `python3 scripts/nonuniform/analyze_nep_collective_probe.py 3065919`.
- All traces and full analysis are under:
  `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3065919/`.
- Traces: `e512_seq2048/cupti/ep8/scaling_ladder__a770m_4b_moe_215b/{healthy,nep}/3065919/cupti_profile/rank-*.json`.
- Summary: `collective_analysis/summary.json`.
- Instrumentation smoke3065843 passed delayed-peer completion, independent
  compute progress, and exact tensor-value checks. Benchmark3065880 failed
  before training on a probe anchor-map bug; fixed in the probe, no runtime edit.
- `AGENTS.md` now requires these records in all future DistOpt benchmarks.
