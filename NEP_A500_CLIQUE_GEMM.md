# a500 topology-controlled expert GEMM comparison

Measured from all 60 rank traces across jobs 2994105 (MBS1) and 2994279
(MBS3), four profiled iterations per trace. GA3, sequence8192, H1024,
expert F640, 128 experts, top-k6, 12 MoE layers. Healthy EP8/8 versus
NEP EP8/6; work remains GPU-proportional. Cross-replica RDMA is verified.

Expert kernels are attributed through CPU `_GroupedLinear` and
`_GroupedLinearBackward` scopes and external IDs, excluding shared experts,
attention, and Mamba. Expected forward/backward kernel counts pass for every
rank and step: full ranks1152/2304; reduced ranks1584/3168 for22experts and
1512/3024 for21experts. No stream-frequency heuristic is used.

Effective throughput is useful routed-expert FLOPs divided by the union of
expert GPU-kernel intervals on a rank, then aggregated by replica role.
Concurrent GEMMs are counted once in elapsed active time. This is not peak
hardware utilization, and it excludes launch gaps, routing, communication,
and activations. Useful work is13.9157/41.7471TF per GPU per iteration for
MBS1/3. Role-average work follows total routed tokens; per-rank random routing
fluctuations are not treated as exact nominal per-rank counts.

| Metric | MBS1 | MBS3 |
|---|---:|---:|
| Healthy expert GPU-active union, ms |22.522|43.371|
| NEP full EP8 expert GPU-active union, ms |23.422|43.550|
| NEP reduced EP6 expert GPU-active union, ms |25.307|46.618|
| Healthy effective TFLOP/s |617.9|962.6|
| NEP full EP8 effective TFLOP/s |594.1|958.6|
| NEP reduced EP6 effective TFLOP/s |549.9|895.5|
| Reduced/full efficiency |92.55%|93.42%|
| Reduced/healthy efficiency |88.99%|93.03%|
| Reduced-minus-full expert GPU-active union, ms |1.885|3.068|
| Same difference divided by MBS, ms |1.885|1.023|
| Logged owner parity, iterations15-24 |82.28%|94.75%|
| Logged NEP-minus-healthy time, ms |64.43|38.77|

Larger dispatches improve absolute GEMM throughput substantially on both
replicas. The within-NEP relative efficiency deficit closes only0.87percentage
points. More experts/GPU remains measurable, but this comparison does not
support attributing the12.47point parity improvement mainly to removing that
deficit. Absolute extra GEMM-active time grows1.885->3.068ms while logged
iteration overhead falls25.66ms. Per unit of batch, the GEMM deficit decreases;
this is not an observed reduction in critical-path iteration time.

The profiled host-step means are395.81/536.08ms (healthy/full NEP) atMBS1
and738.60/810.47ms atMBS3. They differ from post-profile timing windows,
especially MBS1. Therefore subtracting kernel milliseconds from the logged
64.43/38.77ms gaps cannot give a valid causal percentage. Launch/dependency
spacing and communication overlap can still benefit from larger kernels;
these kernel-only measurements do not quantify those effects.

Raw per-rank/per-step metrics and summaries:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/a500_clique_gemm_20260910/`.
Reproduce with `python3 scripts/nonuniform/analyze_a500_clique_gemm.py`.
The cached extraction validates four steps and exact kernel counts per rank;
interval-union tests cover disjoint/overlapping/nested intervals.
