# Backward Normalization Headroom Investigation

Parent: `nep-expert-buckets-20261006`, commit `a6f90f29e`. Training code unchanged.

Hypothesis: concurrent gradient Gather and expert RS consume more SM resources
than the 32-SM backward normalization margin allows, delaying cooperative
RMSNorm admission and propagating waits into native TP/model-EP.

## Controlled Test

- Preserve previous a8b 15-MoE workload, EP16/12, TP2, CP1, MBS2/GA8/S8192,
  DistOpt, Flex/HybridEP, graphs, routing, dense buckets, topology and profiling.
- Fix NEP expert buckets at five; healthy retains five native expert buckets.
- Only vary native `NVTE_BWD_LAYERNORM_SM_MARGIN`: 32 versus 64. Apply each
  to healthy and NEP. Forward normalization margin remains16.
- Two reversed-order repeats; all-rank profiles plus previously authorized
  timing companions; 35 iterations, matched measured steps25-30.
- GB300 only: the previous GB200 case failed unchanged memory-headroom guards.
  Retain segment4 and audit local NVL versus cross-replica RDMA.
- Archive installed TE source and verify instantiated RMSNorm margin before
  training. Confirm actual norm grid changes in each rank's trace afterwards.

## Interpretation

Evidence for causality requires the changed grid to reduce norm launch gaps
and downstream TP/model-EP waits, with early host submission unchanged.
Compare final and ordinary backward, healthy as well as NEP, and total owner
iteration times. Norm runtime can increase with fewer SMs; report that cost.
Do not infer gains by adding residency across ranks or overlapping operations.
Healthy speedup alone is not a NEP parity improvement. If effects are mixed,
report that the margin does not fully explain the remaining overhead.

No production fix will be applied as part of the investigation. If supported,
propose the existing environment setting rather than invasive scheduling code.
