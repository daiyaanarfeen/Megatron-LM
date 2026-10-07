# Expert Count Ablation

## 2026-10-07: Approved Experiment

Isolated from trusted `nep-gather-cta-20261006` at `83c1fd9c0`.
No production `megatron/` changes. The user approved sequence length 8448
instead of 8192 so all three counts support exactly balanced routing at top-k8.

- Expert counts128/256/384, top-k8; matched healthy EP16/16 and NEP EP16/12.
- Saved 15-MoE a8b hybrid/MTP pattern; H4608, latent1152, expertF3072,
  sharedF6144, vocab131072, TP2/CP1/ETP1/PP1, BF16, Flex/HybridEP,
  DistOpt, native MoE CUDA graphs, MBS2/GA8 on both replicas.
- Healthy localBS128/128, GBS256; NEP128/96, GBS224, proportional work.
- Compact placement, ascending owner order, one stream, early delivery,
  prefetch1, norm margins16/32, NEP Gather CTA cap16 fixed throughout.
- Five expert buckets, each covering three successive MoE layers. Benchmark
  startup hook supplies a scaled expert threshold to the unchanged native
  DistOpt layout builder. Dense threshold fixed692000000 elements. Startup
  assertions compare actual expert module and dense parameter groupings.
- Exact randomized balanced routing uses a common96-token unit and seed101.
- 35iterations, matched measured25-30; all-rank Nsight and AG/RS/Gather/
  redistribution event records plus previously authorized timing companions.
- Two reversed repeats. Largest count/NEP passed memory validation in3263307;
  the retry starts128 first to exercise the corrected dense-boundary control.
  Eight GB300 nodes, segment4, one NVL domain per replica, cross-replica RDMA.
  Existing16GiB reserved-memory and8GiB free-memory margins remain mandatory.

Within-count initialization, routing, native payload, transport and repeat
checks remain enforced. Across counts, routing and expert/router weights and
payloads necessarily differ; layer-to-bucket boundaries must remain identical.
This is not an iso-parameter or iso-GEMM-shape sweep: expert count increases
parameter volume and experts/GPU while reducing tokens/expert. Separate these
effects in the phase and delivery analysis; do not attribute all parity changes
to parameter communication.

## Status

Submitted job3263307 from checkpoint `2c6a8602e`. Initially priority pending,
start unknown. CPU routing/layout tests2/2, existing bucket control tests3/3,
six argument-generation checks and shell syntax passed. No GPU results yet.

## 2026-10-07: Dense Boundary Gate Caught a Setup Mismatch

Job3263307 passed eight cases (384 and256, healthy/NEP, profile/companion),
then stopped before128 training. Smaller router weights moved
`decoder.layers.23.mlp.shared_experts.linear_fc1.weight` from dense bucket2
to bucket1 under threshold697279864. Expert groupings were unchanged.
The assertion prevented accepting a confounded128 measurement.

Using the captured shapes and native layout function on CPU gives the common
valid dense threshold interval[687982721,696711040]. Set692000000 for all
cases. Regression preflight now checks actual dense shapes at all three
counts and both DP sizes and proves256/384 layouts (including offsets and
padding) are identical under the old and new thresholds.3/3 tests pass.
No production code or scheduling change. Record failed bucket signatures
before assertion for easier diagnosis. Restart the full sweep in one allocation.

First-repeat preliminary owner timings (measured25-30) retained for reference:

| Experts | Profile healthy/NEP ms | Profile parity | Companion healthy/NEP ms | Companion parity |
|---|---|---|---|---|
|256|3294.549 /3322.450|99.160%|3260.372 /3291.420|99.057%|
|384|3356.925 /3407.505|98.516%|3309.102 /3353.846|98.666%|

384 NEP peak allocated111.30GiB/reserved144.41GiB. Correctness preflight,
memory, routing, payload and transport checks passed for completed pairs.
Both physical replicas were in nvlblk17; separate NCCL cliques enforced
cross-replica RDMA, verified16 healthy/12 NEP pairs. Local reshard groups
were two-rank P2P groups; broadened the offline auditor's previous3-rank
minimum to2 without weakening its P2P/NET transport assertions.

Retry3263820 from `26abd7a1f`, started11:41:58PDT, eightGB300 nodes.

## Final Results: 2026-10-07

Job3263820 COMPLETED0:0 in1h57m39s; all24 cases passed. Actual placement:
full replica theia0145/0146/0148/0149 in nvlblk25; second replica
theia0164/0175/0176/0177 in nvlblk26. All cases verified cross-replica RDMA
(16healthy pairs,12NEP pairs) and local P2P.720 startup rank signatures have
identical expert-module/dense-parameter bucket grouping.360 rank profiles
contain native AG/RS and applicable NEP transfers with completion records.
All cases share the production runtime hash; no production code changed.
Numerical preflight, finite-training, memory, initialization, routing and
payload gates passed. This does not claim bitwise-identical trained weights
between healthy and NEP, which intentionally process different batch totals.

Owner ranks0-11, two reversed-order repeats, measured iterations25-30 after
24 warmup iterations,35 iterations per case. Parity=healthy time/NEP time.
Timing companions use the SAME iteration numbers, not the post-profile window.
CUDA graph coverage is mamba, attention and MoE.

| Experts | Profile healthy ms | Profile NEP ms | Profile parity | Companion healthy ms | Companion NEP ms | Companion parity |
|---|---:|---:|---:|---:|---:|---:|
|128|3220.736|3242.027|99.343%|3168.788|3201.699|98.972%|
|256|3288.450|3324.297|98.922%|3262.663|3280.514|99.456%|
|384|3360.116|3409.184|98.561%|3320.056|3363.714|98.702%|

Repeat parity ranges:128 profile99.139-99.548%, companion98.825-99.119%;
256 profile98.798-99.046%, companion99.049-99.864%;
384 profile98.535-98.587%, companion98.554-98.850%.
Profile-minus-companion NEP gap differences are-11.621,+17.997,+5.410ms
respectively. These include instrumentation AND run variability; they are
not a causal subtraction of profiler cost from individual kernel durations.
The companion result is not monotonic with count, so do not claim a precise
monotonic unprofiled scaling law from these two repeats.

### Profiled Phase Accounting

NEP minus healthy, owner cohort, milliseconds per iteration. Phase spans
include compute, communication, waiting and submission gaps, not pure compute
or transfer service. These explain PROFILED time only, not companion timing.

| Experts | First F | Other F | Non-final B | Final B | Outside F/B | Total gap |
|---|---:|---:|---:|---:|---:|---:|
|128|5.467|3.533|1.436|6.676|4.186|21.291|
|256|9.867|2.054|4.971|12.579|6.390|35.847|
|384|13.534|7.286|6.649|14.835|6.766|49.068|

The phase sum differs from native interval-timer gap by less than0.014ms.

### Delivery Evidence

Device demand-to-ready intervals at parameter consumption, averaged across
owners and measured iterations. These are dependency waits, NOT communication
kernel service time or bandwidth measurements.

| Experts | Param redistribution summed send GB/step | Healthy first-bucket wait ms | NEP first-bucket wait ms | NEP later-four-bucket waits ms |
|---|---:|---:|---:|---:|
|128|6.795|2.128|3.861|0.0261|
|256|13.590|4.021|7.296|0.0259|
|384|20.384|5.951|10.185|0.0258|

Payload is summed source input bytes of `nep_owner_transfer`, excluding
protocol overhead; actual bytes6794772480/13589544960/20384317440. It triples.
Later-bucket consumer waits stay at the marker floor, about0.026ms summed.
This supports timely delivery of later buckets even at384 experts. It does
not establish zero resource contention while delivery overlaps computation.

### Conclusion and Limits

Expert count128->384 does NOT produce the requested>2-3 percentage-point
parity change: the mean spread is0.783pp profiled and0.754pp companion.
This workload retains approximately99% latency parity throughout. The profiled
first-forward and final-backward excesses grow, but do not exhaust later
forward prefetch slack. This is a robustness result, not evidence that count
is a strong parity-breaking factor at this MBS/GA/sequence length.

This changes more than payload in isolation: with fixed top-k/width, total
expert count also changes experts/GPU and tokens/expert. Healthy EP16 holds
8/16/24 true experts per GPU; reduced EP12 holds10-11/21-22/32.128 and256
use the existing nondivisible placement/fixed-width dispatch slots;384 is
divisible. Per-expert routing is exactly balanced; ranks with an extra true
expert naturally receive more expert work. All of these are count-dependent
effects of the approved workload, not an independently varied scheduling knob.

Max allocated GPU memory, healthy/NEP GiB:67.502/78.661 at128,
74.154/95.972 at256,80.805/111.296 at384. Maximum NEP reserved144.451GiB;
all original headroom guards passed.

### Artifacts

- Raw runs: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3263820`.
- `expert_count/summary.json`: timing, repeats, owner/follower/second-replica
  phases and per-rank delivery deadline records.
- `expert_count/{128,256,384}/actual-summary.json`: all cases, measured-window
  variability, phases, readiness records and validated transport manifests.
- `expert_count/expert_count_ablation.png` and `.pdf`: parity with repeat
  ranges plus disjoint profiled phase overhead bars.
- `analysis_source/`: archived analysis scripts for reproduction, using
  `PYTHONPATH=/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/python_plot_tools`
  and `python3 analysis_source/analyze_nep_expert_count.py <run-root>`.
- Talos: https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_expert_count_20261007/
