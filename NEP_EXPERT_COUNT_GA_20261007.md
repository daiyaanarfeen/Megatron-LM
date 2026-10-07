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
