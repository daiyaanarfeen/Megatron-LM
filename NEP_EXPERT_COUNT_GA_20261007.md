# Expert Count at Lower Gradient Accumulation

## 2026-10-07: Requested GA2 and GA4 Sweep

Preserve trusted GA8 commit fc815d606 and all production code. Change only
gradient accumulation and derived batch sizes in the existing expert-count
harness. Run separate matched healthy/NEP sweeps at GA2 and GA4, with
128/256/384 experts and top-k8. Each sweep has two reversed-order repeats,
all-rank profiles plus authorized timing companions: 24 cases per sweep.
The largest expert count runs first for memory validation.

Unchanged: 15-MoE a8b hybrid/MTP, H4608/latent1152/expert F3072,
sequence8448, MBS2, TP2/CP1/ETP1, healthy EP16/16 and NEP EP16/12,
proportional local batches, DistOpt, Flex/HybridEP, CUDA graphs,
five expert/three dense buckets, exact balanced routing, early parameter
delivery, prefetch1, one stream, Gather cap16 CTAs, norm margins16/32.
35 iterations, 24 warmup, measured25-30 in both instrumentation modes.

| GA | Healthy local BS | NEP local BS | Healthy GBS | NEP GBS |
|---|---|---|---|---|
|2|32/32|32/24|64|56|
|4|64/64|64/48|128|112|

Each job requests eight GB300 nodes, segment4, preserving GPU type relative
to GA8. Require each replica inside one NVL domain and audited cross-replica
RDMA. Record actual placement, memory, all-rank AG/RS/Gather/redistribution
records and repeat variability before accepting results. No production
implementation or profiler changes. Keep profile and companion results
separate, using matched iteration indices; no post-profile substitutions.

Question: does the 128-to-384-expert parity drop exceed 2-3 percentage points
at smaller GA, compared with GA8? Also report absolute overhead and phase
breakdown, to distinguish amortization from additional missed delivery deadlines.

## Completed Results

GA2 job3265011 COMPLETED0:0 in1:24:36; GA4 job3265012 COMPLETED0:0
in1:33:53. Sourceabd3a9e17. All48cases validated, no OOM or timeout.
Each job has720identical bucket-group signatures and360complete rank profiles.
GA8/2/4 share NEP runtime hash
faec3d907021672058a975e65fccbf4b4ed1bba326d2ffe98830b998e352f098.
No production diff againstfc815d606. Cross-GA initial weights, native AG/RS
payload fingerprints and NEP transfer bytes match exactly at each expert count.

Placement: GA2 uses nvlblk27/nvlblk32. GA4 occupies nvlblk23 for both
replicas, with existing per-replica NCCL cliques enforcing cross-replica RDMA.
Every case audits16healthy or12NEP cross-replica RDMA pairs and local P2P.
Thus transport is controlled, but physical allocation is not identical across
GAs; every within-GA count sweep shares its allocation. All GPUs are GB300.
Maximum allocated memory is111.296GiB in both jobs, within existing headroom gates.

### Owner Iteration Parity

Ranks0-11; mean of owner times across two reversed repeats and matched
iterations25-30, after24warmup iterations. Parity=mean healthy/mean NEP.
Profiled and authorized timing companions remain separate; no post-profile
timing substitution. GA8 references completed job3263820.

| GA | 128 profile | 256 profile | 384 profile | 128-to-384 drop |
|---|---:|---:|---:|---:|
|2|97.951%|96.917%|95.569%|2.382pp|
|4|98.968%|98.386%|97.782%|1.186pp|
|8|99.343%|98.922%|98.561%|0.783pp|

| GA | 128 companion | 256 companion | 384 companion | 128-to-384 drop |
|---|---:|---:|---:|---:|
|2|97.553%|96.739%|95.731%|1.822pp|
|4|98.722%|99.421%|98.161%|0.561pp|
|8|98.972%|99.456%|98.702%|0.270pp|

The count sensitivity grows at smaller GA. Only the GA2 PROFILED mean
exceeds the requested2pp threshold; its companion mean does not. Do not
claim a robust>2-3pp unprofiled effect. Companion curves atGA4/8 are not
monotonic: the256-expert point must not be omitted or smoothed away.

### Absolute Timing and Variability

| GA | Experts | Profile H/N ms | Companion H/N ms | Profile gap ms | Companion gap ms |
|---|---|---|---|---:|---:|
|2|128|832.834/850.259|817.872/838.386|17.425|20.514|
|2|256|856.558/883.803|844.216/872.677|27.245|28.461|
|2|384|883.620/924.589|870.337/909.149|40.969|38.812|
|4|128|1643.925/1661.060|1620.691/1641.668|17.135|20.978|
|4|256|1673.900/1701.361|1665.313/1675.016|27.461|9.703|
|4|384|1723.129/1762.207|1703.974/1735.900|39.078|31.926|

Profile parity repeat rangesGA2:97.604-98.298,96.600-97.236,95.097-96.045%;
companion97.493-97.613,96.511-96.966,95.576-95.886% for128/256/384.
GA4 profile98.844-99.092,98.340-98.432,97.540-98.026%;
companion98.641-98.804,99.035-99.807,98.089-98.232%.
The128-to-384 drop in each paired repeat isGA2 profile3.201/1.558pp,
companion2.037/1.607pp; GA4 profile1.553/0.818pp,
companion0.551/0.572pp. Two repeats are not a statistical significance test.

Profile-minus-companion GAP differences at128/256/384 areGA2
-3.089/-1.216/+2.156ms andGA4 -3.842/+17.758/+7.152ms. These include
instrumentation and run variability, not an exact subtraction of profiler cost.

### What the Profiles Explain

Profiled NEP-minus-healthy owner phase durations, milliseconds. These are
disjoint phase spans, NOT isolated compute or transfer service times.

| GA | Experts | First F | Other F | Nonfinal B | Final B | Outside F/B |
|---|---|---:|---:|---:|---:|---:|
|2|128|4.975|0.679|-0.090|8.548|3.283|
|2|256|8.899|0.168|0.025|12.738|5.405|
|2|384|13.572|1.570|2.221|17.579|6.163|
|4|128|6.287|2.941|2.036|8.983|-3.112|
|4|256|9.816|1.087|3.249|12.471|0.810|
|4|384|13.002|2.235|3.261|22.384|-1.827|

AtGA2, count128->384 adds23.544ms total profiled excess: firstF+8.597ms,
finalB+9.031ms, and remaining phases+5.916ms. About75% of the added gap
lies in firstF/finalB. The absolute gaps atGA2/4 remain similar at each count
(17/17ms,27/27ms,41/39ms), but GA2 iterations are approximately half as long.
This supports reduced amortization, not a new collapse of later-forward hiding.

| GA | Experts | NEP first-bucket demand wait ms | Later-four-bucket wait ms |
|---|---|---:|---:|
|2|128|3.803|0.02636|
|2|256|7.264|0.02641|
|2|384|10.192|0.02640|
|4|128|3.814|0.02581|
|4|256|7.240|0.02582|
|4|384|10.130|0.02590|

These device demand-to-ready intervals are not transfer wire times. Later
deliveries remain at the marker floor, as atGA8; no measured later-bucket
deadline failure. First-bucket wait rises with payload but barely changes
withGA. This does not prove zero contention or assign every phase millisecond
to communication. Expert count also changes tokens/expert and experts/GPU.

### Artifacts

- Raw jobs under `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/`.
- Each job's `expert_count/summary.json` retains timings, phases, delivery waits
  and repeats; per-count `actual-summary.json` and `work-controls-audit.json`
  retain per-rank records and transport/payload checks.
- All-rank Nsight reports/SQLite and GPU event records remain in each case.
- `expert_count/expert_count_ablation.png` and `.pdf` show correctly labeled
  GA2/4, local/global batch sizes, bucket counts and phase excesses.
- Plot publication: `https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_expert_count_ga_20261007/`.
