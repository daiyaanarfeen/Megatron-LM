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
