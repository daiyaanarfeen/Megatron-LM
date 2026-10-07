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
  DistOpt layout builder. Dense threshold fixed697279864 elements. Startup
  assertions compare actual expert module and dense parameter groupings.
- Exact randomized balanced routing uses a common96-token unit and seed101.
- 35iterations, matched measured25-30; all-rank Nsight and AG/RS/Gather/
  redistribution event records plus previously authorized timing companions.
- Two reversed repeats, largest count/NEP first for memory validation.
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

Harness preparation; no submitted run yet.
