# Strong Ordinary-Microbatch Sensitivity

## Final Status: Three Authorized Sweeps Complete

2026-09-15 after10:30 PDT. Jobs3063804,3064000,3064218 allCOMPLETED0:0,
elapsed27m04s,29m14s,28m25s. No outstanding study jobs and no further launches.
All24 cases trained60 iterations. All360 rank profiles /3600 measured
rank-iterations passed schema/count/clock checks and finite-loss/no-skip/no-NaN
checks. Runtime93d313df4, compiled helper and each frozen harness SHA validated
unchanged. Per-case physical domain containment, healthy/full fabric and
software-clique cross-replica RDMA passed. No runtime edits or remote pushes.

All results use ranks0-5 owner mean per iteration, then mean/sampleSD over
iterations40-49. Pre29-38 andpost51-60 diagnose drift/distortion, not correction.
MBS2,GA3,proportional EP8/8 healthy versusEP8/6 NEP, GBS96/84, local replica
samples48/48 versus48/36, TP/CP/PP/ETP1, DistOpt+Flex/HybridEP throughout.
Every comparison has its own matched healthy/NEP workload. Each sweep uses one
GB200 allocation, segment4. Original anchor H1280/F768/sharedF1536/E128/topk6,
seq8192,15MoE layers with original hybrid pattern. Knobs below are independent
changes from that anchor; no GA, dispatcher, routing or profiling changes.

| Job / case | Healthy mean +/-SD ms | NEP mean +/-SD ms | Measured owner parity | Pre/post parity |
| --- | ---: | ---: | ---: | ---: |
|3063804 anchor start|702.47 +/-2.80|732.14 +/-2.56|95.95%|96.19/96.28%|
|3063804 seq2048|339.66 +/-4.95|390.96 +/-5.49|86.88%|86.38/85.92%|
|3063804 seq1024|350.43 +/-5.37|380.85 +/-1.51|92.01%|90.87/92.52%|
|3063804 anchor end|702.93 +/-3.07|729.99 +/-13.86|96.29%|98.79/96.66%|
|3064000 anchor start|692.33 +/-2.99|725.31 +/-3.21|95.45%|93.54/95.07%|
|3064000 E512|927.30 +/-5.18|1081.41 +/-7.70|85.75%|87.53/86.23%|
|3064000 E512+seq2048|753.59 +/-8.94|923.02 +/-12.10|81.64%|78.61/81.32%|
|3064000 anchor end|698.06 +/-3.73|734.72 +/-10.92|95.01%|94.51/95.41%|
|3064218 anchor start|702.46 +/-3.04|727.00 +/-6.12|96.62%|95.84/96.70%|
|3064218 H320|589.93 +/-2.75|608.12 +/-6.06|97.01%|97.11/97.43%|
|3064218 H320+E512|799.42 +/-9.32|931.29 +/-11.69|85.84%|82.94/87.77%|
|3064218 anchor end|702.26 +/-6.13|729.39 +/-3.63|96.28%|95.45/96.42%|

### Corrected Ordinary-Work Interpretation

Owner-only phase partitions do not identify causes. Replicas are independent
until final gradient synchronization; slower reduced ordinary work can cause
full owners to wait in finalB/tail. Earlier journal language describing negative
ordinary attribution from owner spans alone is explicitly superseded here.

E512 reduced ranks lose24.17ms between firstF end and finalB start:18.45ms
additional laterF +5.04ms earlierB +0.68ms other transitions. They move from
18.52ms ahead after firstF to5.64ms behind before finalB, and end finalB25.33ms
later. E512+seq2048 loses29.79ms:21.82ms laterF +10.97ms earlierB -3.00ms
other transitions; reduced moves14.44ms ahead to15.36ms behind before finalB.
Both original E512-job anchors retain/extend reduced's lead. Healthy second
replicas remain ahead, not the same delayed pattern. This is affirmative
progress evidence for reduced ordinary work contributing to final readiness.
It is not evidence that all154-169ms owner overhead comes from that mechanism.

The H320+E512 point is more limited: reduced loses12.85ms of an initial13.95ms
lead, remaining1.10ms ahead at finalB start and6.30ms ahead at its end on average.
Its laterF/earlierB spans add5.03/5.65ms relative to full owners over twoMBs;
remaining2.17ms loss is in transitions. Compared with healthy, full-owner
laterF/earlierB totals add3.16/1.51ms, but finalB adds73.42ms. It does not
establish the same mean late-reduced readiness bottleneck as the E512 sweep.
Neither event partitions nor cohort averages prove exact NCCL critical waits.

### Host-Work Evidence

Numbers below are per ordinary microbatch, averaged across twoMBs and10steps.
CPU wall/thread are measured separately; tiny thread>wall differences arise
from timer boundaries. GPU elapsed sums include dependencies/host spacing,
not pure kernel busy time. Full and reduced here are within the SAME NEP run.

| Case/cohort | LaterF CPU wall/thread ms | LaterF GPU elapsed sum ms | EarlierB CPU wall/thread ms | EarlierB GPU elapsed sum ms | EarlierB launch-done-ahead | EarlierB start lead ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
|E512 full|32.41/32.39|32.94|65.24/65.25|19.23|100.0%|40.80|
|E512 reduced|41.83/41.83|42.38|81.26/81.27|22.76|99.8%|30.39|
|E512+seq2048 full|33.63/33.58|34.19|65.50/65.24|22.51|76.7%|10.62|
|E512+seq2048 reduced|40.98/40.98|41.52|76.89/76.91|28.30|67.5%|5.36|
|H320+E512 full|33.76/33.74|34.32|67.76/67.77|12.33|99.4%|20.59|
|H320+E512 reduced|39.46/39.46|39.98|77.00/77.01|15.40|97.6%|16.53|

All these laterF submissions have essentially0% launch-done-ahead. Added
reduced F thread work is close to added grouped GPU elapsed range time,
consistent with host-paced grouped forward execution. EarlierB has more
submission lead; shortening sequence reduces it materially. Do not equate
these observations with exact CPU-caused GPU idle or GEMM efficiency.
All ranks retain90 grouped calls per direction/iteration; real experts/GPU
change16->64 full and21/22->85/86 reduced. Individual kernel counts unmeasured.

H320+E512 keeps the original logical expert weight volume (Hquarter*Equadruple),
but it does NOT keep parameter tensor counts or host work fixed. Owner NEP
gradient-stage CPU wall/thread grows10.18/10.19ms in its anchor to39.60/39.60ms;
parameter-redistribution CPU grows8.78->24.27ms; optimizer CPU is61.30ms versus
healthy47.57ms. Gradient-stage GPU elapsed sums6.48->17.69ms and redistribution
9.55->17.63ms are envelopes, not transfer-only service. These support
count-sensitive synchronization-host overhead despite equal expert byte volume.
Exact attribution of the73.42ms extra finalB span remains unresolved.
Reduced grad-stage CPU starts are nearly aligned(-0.10..+0.78ms relative tofull)
and GPU stage ends remain earlier(-0.16..-6.35ms), not a large late-stage pattern.

### Controls, Limits And Artifacts

Native actual buckets: anchor3dense/4expert; E512 andH320+E5122dense/7expert;
H320-only3dense/4expert. NEP8logical owner groups throughout. Bucket sizes are
in summary.json; equal counts do not imply equal sizes. Same E512 across
sequence has identical signatures. This is not an iso-bucket expert-count study.
H320 keeps Mamba40heads*64=2560 inner width; scan work is NOT quartered.
Attention KV128/headcounts, sharedF1536,F768,seq8192 unchanged in hidden sweep.
E512 changes router width and per-expert token distribution as derived effects.
H320E512 max-owner packed bytes are slightly lower from ceil(E/6) rounding;
logical expert total is identical, not every rank's padded buffer size.

H320E512 parity moves82.94->85.84->87.77 across windows: low parity is robust,
but exact85% is not invariant and no profiler correction is applied. Absolute
cross-host arrival comparisons assume clocks not independently verified here.
Local progress differences cancel constant offsets; epoch iteration-start
cohort differences are below0.1ms in recorded data. Per-position staging samples
and SDs are retained; staging ordinals are call order, not exact collective IDs.
No claim of total GPU starvation, pure GEMM throughput or exact causal-ms sum.

Root: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/.
3063804/ordinary_strong_analysis,3064000/ordinary_experts_analysis,
3064218/ordinary_hidden_analysis each contain summary.json, per-rank samples,
additive_ordinary_breakdown.json and final_validation.json. Jobs3064000 and
3064218 also have ordinary_progress_analysis/summary.json with CPU/GPU boundary
progress and all8 staging positions. Raw profiles are TAG/cupti/ep8/
scaling_ladder__a770m_4b_moe_215b/{healthy,nep}/JOB/cupti_profile/rank-N.json.
Frozen code-only harnesses are experiments/nep_ordinary_{strong,experts,hidden}_20260915.
Each ordinary_manifest.json records accepted runtime manifest plus harness SHA.
Run/analyze scripts have matching names in scripts/nonuniform. Historical
preparation/results below preserve chronology including the causal correction.

## 2026-09-15: Preparation

Authorized stronger sequence-only sweep: anchor healthy/NEP, sequence2048
NEP/healthy, sequence1024 healthy/NEP, anchor-end NEP/healthy. GA3 and MBS2
remain fixed. EP8/8 versus EP8/6, proportional GBS96/84, replica samples
48/48 versus48/36. H1280, expertF768, sharedF1536, 128 true experts, topk6,
15 MoE layers, TP/CP/ETP/PP1, DistOpt and Flex/HybridEP unchanged.
Sequence changes reduce tokens per microbatch to one quarter / one eighth;
parameter payloads and requested gradient bucket settings remain fixed.
Actual bucket signatures will be audited, not assumed identical to requests.

Accepted runtime remains the frozen snapshot of93d313df4 at
/home/darfeen/Megatron-LM-nep-sensitivity-accepted-20260914.
All tracked runtime hashes validated against sensitivity_manifest.json.
No runtime edits. New isolated harness:
/home/darfeen/Megatron-LM/experiments/nep_ordinary_strong_20260915.
New sbatch wrapper: scripts/nonuniform/run_nep_ordinary_strong_20260915.sh.

Four nodes, segment4, either GB200 or GB300, same hardware within all pairs.
Distinct software NVLink cliques enforce cross-replica RDMA. Named Enroot cache.
Each case60 iterations, event profiles all ranks40-49, surrounding timing windows
29-38 and51-60, six-minute case watchdog. No profiler disabled. Total35minutes.
All large artifacts in Lustre soft_clique_baseline/JOB/TAG, unchanged schema.

Goal: seek about85% owner parity and determine whether ordinary later-forward /
earlier-backward penalties explain it. Negative findings and remaining submission
headroom will be reported. Do not conflate synchronization tail with ordinary
phase slowdown or GPU event envelopes with pure GEMM kernel residency.
Parent owns separate expert-width synchronization sweep; no duplicate cases.

## 2026-09-15 08:59 PDT: Job3063804 Allocated

Submitted08:58:38, allocated08:59:08 on gb200-backfill, nodes
lyris[0262,0264-0266], segment4, deadline09:34:08. Both replicas contained
in physical nvlblk15; software cliques1000/1001 enforce cross-replica RDMA.
No blocker. Exact-one-argument launcher checks passed for all three variants.
Harness SHA manifest: experiments/nep_ordinary_strong_20260915/ordinary_manifest.json.
Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3063804/.
Offline analyzer: scripts/nonuniform/analyze_nep_ordinary_strong_20260915.py,
output3063804/ordinary_strong_analysis; parent may reuse event files separately.

## 2026-09-15 09:21 PDT: Sequence Pair Results, Ordinary Attribution Negative

Initial anchor owner parity95.95%, pre96.19/post96.28; healthy702.47+/-2.80ms,
NEP732.14+/-2.56ms. Seq2048 parity86.88%, pre86.38/post85.92;
healthy339.66+/-4.95ms, NEP390.96+/-5.49ms. Seq1024 parity92.01%,
pre90.87/post92.52; healthy350.43+/-5.37ms, NEP380.85+/-1.51ms.
All measured values same iterations40-49 across ranks0-5. Final anchor pending.

Owner NEP-minus-healthy temporal deltas (ms), ordinary phases summed over two
microbatches, finalB one microbatch:

| Case | LaterF | EarlierB | FinalB | LastB to optimizer | FirstF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anchor start |4.91|0.27|2.32|2.19|15.23|
| Seq2048 |2.04|-3.80|20.92|13.80|14.81|
| Seq1024 |-11.84|2.37|25.49|6.20|7.48|

The low score atseq2048 is NOT a successful ordinary-phase ablation: most
positive change occurs in sync-adjacent regions. Reduced earlierB CPU submission
lead falls62.37->4.80->4.51ms; fraction finishing before the corresponding GPU
range starts falls100%->91.0%->82.6%, but owner ordinary spans do not substantially
increase. These ranges do not establish exact GPU starvation or individual GEMM
efficiency. Positive reduced-vs-healthy-second-replica spans alone do not establish
owner critical-path loss. All cases preserve actual bucket signatures so far.

## 2026-09-15 09:26 PDT: Authorized Follow-up Expert-Count Sweep

Parent authorized E128->512 and E512+seq2048 after current job finishes, with
original anchors start/end. New isolated harness experiments/nep_ordinary_experts_20260915;
wrapper scripts/nonuniform/run_nep_ordinary_experts_20260915.sh. No runtime changes.
64 real experts/full GPU and85/86 reduced, versus16 and21/22. HybridEP virtual
slot machinery supports this nondivisible placement. Topk6, MBS2, GA3 unchanged.
Expert parameter payload quadruples; tokens per expert quarter atseq8192 and
fall to1/16 of original atseq2048. Router output width also quadruples. Actual
buckets may change as a derived consequence; capture and report them explicitly.
Eight sequential cases, four nodes,40minute limit,6minute case watchdog.

An alternative narrow-expert harness was mechanically copied but never submitted
or used. It is superseded by the explicitly authorized E512 follow-up, not a
failed experiment. No extra MBS or shared-width change is being launched.

## 2026-09-15 09:29 PDT: First Sweep Completed, Expert Sweep Running

3063804 COMPLETED0:0,27m04s, final anchor96.29% owner parity
(pre98.79%, post96.66%). All eight cases60 iterations,120 all-rank profiles,
1200 measured rank-iterations. Actual bucket/config validation passed all pairs.
The ordinary-phase negative conclusion above remains: no substantial owner
laterF/earlierB penalty despite seq2048 iteration parity86.88%.

3064000 submitted09:29:18, allocated09:29:31 on the same four GB200 nodes
lyris[0262,0264-0266], segment4, deadline10:09:31. Fresh anchors retained.
All six variant/world-size launch tests passed. Harness manifest in
experiments/nep_ordinary_experts_20260915/ordinary_manifest.json.
Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3064000/.
Analyzer: scripts/nonuniform/analyze_nep_ordinary_experts_20260915.py,
outputs ordinary_experts_analysis, accepts payload-derived bucket changes only
between E128/E512 and requires equal signatures across sequence at fixed E.

## 2026-09-15 09:54 PDT: Expert-Count Pair Results

E512 owner parity85.75% (pre87.53/post86.23), healthy927.30+/-5.18ms,
NEP1081.41+/-7.71ms. E512+seq2048 parity81.64% (pre78.61/post81.32),
healthy753.59+/-8.94ms, NEP923.02+/-12.10ms. Anchor95.45% measured.

Owner additive delta ms, ordinary directions summed over two microbatches:

| Case | LaterF | EarlierB | FirstF | FinalB | LastB to optimizer |
| --- | ---: | ---: | ---: | ---: | ---: |
| E512 | -1.25 |0.64 |50.52 |51.87 |39.25 |
| E512+seq2048 |13.23 |-21.34 |60.17 |68.63 |20.19 |

NOT an85%-ordinary-phase success. Low owner iteration parity is robust across
recorded/surrounding windows but synchronization-adjacent spans dominate.
Reduced earlierB grouped CPU wall/thread ms rises81.26/81.27 versus full
65.24/65.25 at E512; GPU elapsed operation sums22.76 versus19.23ms, not pure
kernel service. Reduced launch-done-ahead fraction99.8%, start lead30.39ms.
At E512+seq2048 reduced CPU wall/thread76.89/76.91 versusfull65.50/65.24;
GPU elapsed sums28.30 versus22.51; launch-ahead67.5% reduced /76.7% full,
reduced start lead5.36ms. Small thread-minus-wall differences reflect separately
sampled timer boundaries; these values cannot identify exact GPU starvation.

Important attribution boundary: native and NEP register_grad_ready synchronize
only is_last_microbatch. Increased ordinary work on the reduced replica can
contribute to its final readiness delay, whose cost on full owners appears at
finalB/tail. Lack of large owner ordinary spans does not by itself rule this out.
The current event decomposition has not separated that wait from communication.

## 2026-09-15 10:00 PDT: Final Authorized Hidden-Width Sweep Prepared

Parent authorized one final sweep: original anchor H/N, H320-only N/H,
H320+E512 H/N, original anchor-end N/H. MBS2,GA3,seq8192,F768,sharedF1536,
topk6,attention KV128 and12heads,Mamba40heads remain unchanged.
Selected H320 rather than additionally running F192: fewer expert/shared and
projection FLOPs with more experts/CPU bookkeeping, while expert weight volume
H320*E512 equals originalH1280*E128 (bias disabled). H320 alone quartersexpert
weight volume. Router matrix H320*E512 also retains original weight volume,
but its output expert axis quadruples. Dense parameters and buckets may change.

Correction explicitly preserved: Mamba mamba_num_heads40 fixes d_inner=2560
with head_dim64 independently ofH. This does NOT quarter SSM scan work or head
dimension. Projection widths change; attentionKV128 avoids H/12 derivation.
TP/CP/PP/ETP1 avoids additional sharding divisibility constraints. No runtime
change or extra independent shape change authorized/needed by inspected code.

Harness experiments/nep_ordinary_hidden_20260915, wrapper
scripts/nonuniform/run_nep_ordinary_hidden_20260915.sh. Eight cases35minutes,
same6minute watchdog and all-rank40-49 profiles. Do not launch any fourth sweep.

## 2026-09-15 10:05 PDT: E512 Complete, Final Hidden Sweep Running

3064000 COMPLETED0:0 in29m14s. Final anchor95.01% (pre94.51/post95.41).
All120 profiles,1200 measured rank-iterations, finite losses/no skips/NaNs,
runtime/helper/harness hashes and all-case fabric/RDMA validation passed.
Native rank0 buckets E128:3dense/4expert; E512:2dense/7expert; NEP8logical
owner sync groups in both. E512 vsE512seq2048 signatures identical. Derived
payload/bucket change is intentional and not an iso-bucket comparison.

3064218 submitted10:01:31, pendingPriority initially, allocated10:01:54
onGB200 lyris[0002,0004,0016-0017], segment4. Observedrunning10:05.
Artifact root soft_clique_baseline/3064218, tagsanchor_start,h320,h320_e512,
anchor_end. New analyzer analyze_nep_ordinary_hidden_20260915.py writes
ordinary_hidden_analysis. Same accepted snapshot and profiling protocol.

## 2026-09-15 10:09 PDT: Progress Audit Qualifies Causal Interpretation

Earlier claims of an ordinary-phase negative result apply to DIRECT full-owner
temporal spans, not causal exclusion of reduced ordinary work. The code gates
grad synchronization on is_last_microbatch; slower reduced ordinary work can
make full owners wait in finalB/tail rather than in their own ordinary spans.

New analyzer scripts/nonuniform/analyze_nep_ordinary_progress_20260915.py writes
3064000/ordinary_progress_analysis/summary.json. It retains rank-local elapsed
boundaries and epoch-time differences separately. Fixed host-clock offsets
cancel from local elapsed differences; absolute cross-host arrival ordering
still assumes clock synchronization not independently proven by these traces.
Recorded cohort iteration-start differences are all below0.1ms, consistent
with aligned starts but not proof of synchronized host clocks.

Reduced minusfull owner mean elapsed-progress difference(ms): negative=ahead.

| Case | FirstF end | FinalB start | FinalB end | Last grad-stage end |
| --- | ---: | ---: | ---: | ---: |
| Anchor start |-12.17|-17.73|-12.55|-12.78|
| E512 |-18.52|5.64|25.33|23.62|
| E512+seq2048 |-14.44|15.36|9.24|6.78|
| Anchor end |-12.51|-25.13|-18.71|-18.89|

E512 loses24.17ms of lead between firstF end and finalB start: additional reduced
laterF18.45ms +earlierB5.04ms +other transitions0.68ms. E512+seq2048 loses29.79ms:
laterF21.82 +earlierB10.97 -other3.00ms. Both anchors instead extend the lead.
Healthy second replicas reach finalB start earlier than full owners by46.11ms
forE512 and74.58ms forE512seq2048; those are separate healthy progress asymmetries,
not assumed NEP wait savings. Mean reduced finalB-end lag E51225.33+/-9.87ms,
E512seq2048 9.24+/-16.26ms; last-stage lag23.62+/-8.71 and6.78+/-16.06ms.
The latter has large iteration variability; do not claim every iteration late.

This supports increased reduced ordinary duration consuming the initial lead
and contributing to final readiness. It does NOT partition the154-169ms total
owner iteration overhead into communication service versus readiness waits,
and event ranges alone cannot identify exact NCCL waits or pure GPU efficiency.

## 2026-09-15: Explicit Causal Correction And Staging Audit

Do NOT conclude ordinary reduced-replica overhead is absent merely because
owner ordinary spans are flat/shorter. E512 reduced-minus-full progress is
already positive atMB2 forward end, before final backward gradient sync.
The loss of initial lead is accounted for by measured ordinary durations.
The earlier negative result is only that extra owner wall time resides mainly
in sync-adjacent intervals; those temporal intervals are NOT causes.

Progress analyzer now retains all8 CPU/GPU grad-stage start/end positions in
submission order. Ordinals are recorded call order, not asserted bucket IDs;
they reflect staging/scheduler timing, not exact NCCL submission/peer-ready
timestamps. CPU thread/wall and GPU elapsed operation envelopes are kept
separate. The observed additional ordinary work is supported; its precise
critical-path share within total NEP overhead remains bounded, not assigned
all154-169ms. No further sweep beyond running3064218 is authorized/planned.
