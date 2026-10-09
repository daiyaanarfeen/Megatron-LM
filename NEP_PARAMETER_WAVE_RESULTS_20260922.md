# Cached NEP Parameter Waves: Measured Result

## Decision

Keep the isolated candidate `nep-param-waves-20260922`, commit `8f840ff72`.
The current/trusted branches were not changed. Runtime scope:15 added lines and
one replaced line, only in cached DistOpt parameter-transfer plan construction.
Gradient scheduling, group creation, ownership, payloads, streams, buffers,
dispatcher and optimizer mathematics are unchanged.

EP16/12 owner first forward improves17.462ms and iteration improves16.429ms.
This is a useful improvement, not complete parity or an irreducibility claim.

## Validation Ladder

1. Job3135093, one GB200 node:88 unit tests pass; four-GPU real-plan parameter
   redistribution passes divisible/nondivisible placement, shared followers,
   fresh parameter values, bounded buffer reuse, async/sync and force-sync cases.
   Existing native EP/AG/RS/prefetch correctness test also passes. Black/isort run.
2. Job3135135, four GB200 nodes/segment2:EP8/6, four MoE layers,H4608/F3072,
   latent1152,E512,topk6,sequence1024,MBS2/GA2,TP2/CP1. Current and candidate
   both complete25 iterations with all-rank Nsight plus CUDA-event records.
   Owner measured15-20: firstF42.090->41.314ms, iteration155.786->155.208ms.
   OtherF4.497->4.526, nonfinalB7.716->7.713ms. FinalB rises2.527ms while
   outside-F/B falls2.355ms; do not claim every small-test phase improved.
   All2580 captured collective calls have identical payloads/counts across
   versions; follower7 order changes2/3/4/5->3/4/2/5 as intended.
3. Job3135236, eight GB300 nodes/segment4: healthy/current/candidate sequentially
   on the same allocation. All35 iterations/case finite with zero skipped steps,
   all32/28/28 rank event profiles and Nsight SQLites present. No timeout.

The four-GPU test checks exact copied parameter values. Finite training is not a
claim of bitwise equivalence: the small runs' printed losses differ slightly
after early iterations; the validation here does not establish bitwise trajectory
identity for native nondeterministic training.

## Hard Workload and Placement

- Original hard a8b latent hybrid configuration,15 MoE layers in its retained
  hybrid/MTP pattern,H4608,latent1152,F3072,E512,topk6,sequence8192.
- MBS1/GA2 on both replicas,TP2/CP1/ETP1/PP1. Healthy EP16/16,GBS32,local16/16;
  NEP EP16/12,GBS28,local16/12: proportional local batch allocation.
- Flex/HybridEP,DistOpt,native MoE CUDA graphs,uniform padded routing;
  NVTE backward margin32 in all cases.
- Full replica:theia0028-0031,NVL domain18. Second replica:
  theia0275/0278/0279/0287,NVL domain32; NEP uses the first three of these nodes.
  Replica-specific MNNVL cliques1000/1001 retained.
- Same requested DDP count8; inherited actual expert bucket counts healthy5,
  NEP8. This is not an iso-bucket healthy comparison. Crucially both NEP versions
  use the identical8-bucket layout and exact same collective payloads.
- Profiled iterations20-27 after19 warmup iterations. Owner cohort ranks0-11;
  each sample averages that cohort within one iteration, then averages8 steps.

## Matched Profiled Phases

Times are milliseconds. Parity is healthy owner time / NEP owner time.

|Phase|Healthy|Current ccb4b9726|Waves8f840ff72|Candidate-current|
|---|---:|---:|---:|---:|
|First forward|98.030|132.590|115.128|-17.462|
|Other forward|79.641|79.745|79.817|+0.072|
|Nonfinal backward|120.516|120.307|120.593|+0.286|
|Final backward|134.430|155.917|155.721|-0.196|
|Outside forward/backward|58.918|72.534|73.404|+0.870|
|Iteration|491.535|561.093|544.664|-16.429|
|First-forward parity|100%|73.94%|85.15%||
|Iteration parity|100%|87.60%|90.25%||

Iteration sample SD:5.071/1.859/1.383ms, respectively; first-forward
SD:5.176/.202/.565ms. Healthy has one slower measured iteration24(503.907ms);
all8 matched iterations are retained, no cherry-picking. The absolute parity
therefore inherits healthy variability. The candidate's total time is below the
reference's on every measured step, with no overlap between their observed
iteration-time ranges(reference558.992-564.269,candidate542.578-546.099ms).

Measured-log minus post-profile mean:healthy14.991ms,current12.902ms,waves11.636ms.
These window differences include runtime variance, not solely profiler overhead.
The post-window current->candidate change is547.886->532.714ms(15.171ms), provided
only as a separate sensitivity check. It is NOT substituted into the phase table.

Peak allocated/reserved memory for both NEP versions:116.394/134.453GiB,
unchanged. Healthy peak allocated:60.707GiB.

## Why It Helped

Same global cached order is computed before participant filtering:
`[0,3,6,9], [1,4,7,10], [2,8], [5,11]`. There are no wave barriers.
Independent owners can now make progress without the old follower-to-follower
chain. Each GPU still has one redistribution stream and serialized broadcasts.

First-forward diagnostics, mean per owner/iteration:

|Diagnostic|Current|Waves|
|---|---:|---:|
|Native expert AG kernel union|56.811|57.165|
|NEP redistribution kernel union including staging|39.572|18.098|
|Redistribution broadcast residency|33.462|11.616|
|Broadcast prefix before last peer kernel starts|30.973|8.293|
|Broadcast remainder after last peer kernel starts|2.490|3.323|
|Native model-EP exclusive residency inside forward graphs|25.284|12.764|
|HybridEP ag_nvl prefix before last peer kernel starts|18.279|5.555|

These are overlapping diagnostics, NOT additive contributions to the17.462ms
first-forward saving. A peer-start prefix is not pure transport time and does
not prove bandwidth saturation. Broadcast post-peer-start time rose0.834ms,
so this did not make the link intrinsically faster: it primarily removed waiting
and the resulting late arrival to native model-EP operations.

The final parameter bucket is ready across all16 full-replica ranks at100.269ms
before versus82.638ms after, relative to rank0 first-forward start. Its17.631ms
advance closely matches the17.462ms phase saving. Before,rank11 was always last;
after,rank5 or11 is last. This is a GPU-completion readiness measurement, not
a host-enqueue interval.

All17952 captured collective calls across28 ranks and8 steps match in operation,
participants,payload and count. Trace-derived broadcast order matches the proposed
waves. All128 full-replica rank/step samples have nonoverlapping NEP broadcast
kernels and one staging stream. An initial overly strict staging-range assertion
was corrected: Nsight cuda-sw reports1-3us overlap between consecutive copy-kernel
activity ranges on the SAME stream in both versions. Broadcast non-concurrency
is checked separately; no runtime change was made in response.

## Artifacts

All large outputs are under `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs`:

- `nep_buffer_cleanup/3135093`:unit and four-GPU test logs/event traces.
- `nep_cleanup_ab/3135135`:small A/B, `cleanup_phase_analysis.json`,
  `wave_validation.json`.
- `soft_clique_baseline/3135236`:hard cases and all-rank raw profiles,
  `cleanup_phase_analysis.json`, `wave_validation.json`,
  `ag_redistribution_analysis/overlap_results.json`, per-rank delivery caches.

Reproduction wrappers:`run_nep_buffer_cleanup_unit.sh`, `run_nep_cleanup_ab.sh`
with `CLEANUP_CASES='ag_overlap_margin param_waves_margin'`, and
`run_nep_hard_cleanup_ab.sh` with
`HARD_CASES='healthy_margin ag_overlap_margin param_waves_margin'`.
Analysis:`analyze_nep_cleanup_ab.py 3135236 --hard`, then
`analyze_nep_ag_red_overlap.py 3135236 --labels healthy_margin ag_overlap_margin param_waves_margin`.
