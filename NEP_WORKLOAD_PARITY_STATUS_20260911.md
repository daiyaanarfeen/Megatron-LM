# Workload parity and mechanism status

Updated 2026-09-11. Parity is healthy / NEP full-owner iteration time.
All percentages compare matching windows within each healthy/NEP pair.
No profile phase is used to explain a different unrecorded window's milliseconds.

## Topology-controlled workload results

Each replica is within a physical NVL domain. Separate software MNNVL cliques
force cross-replica RDMA; this is an emulated cross-domain transport experiment,
not a claim that the replicas occupy distinct physical domains. The first three
model rows use full PyTorch CPU+CUDA profiles, with substantial instrumentation
effects. The latest a770 pair uses low-overhead targeted CUDA-event scopes, not
a complete kernel trace. Do not compare profiler types as if identical.

| Model | Config change | Profiled owner parity | Post-profile owner parity | Jobs |
| --- | --- | --- | --- | --- |
| Ladder a315m/1b | MBS4 GA3, original only | 90.62% | 97.25% | 2993769 |
| Ladder a500m/2b | MBS1 -> 3, GA3 fixed | 73.82% -> 91.13% | 82.28% -> 94.75% | 2994105 / 2994279 |
| a8b/120b latent MoE | MBS1 GA8, original only | 93.14% | 93.89% | 2994648 |
| Ladder a770m/4b, latest | MBS1 -> 2, GA3 fixed | 88.36% -> 95.76% | 88.71% -> 95.27% | 3007029 |

The older full-profile a770 pair2995392/2995524 gave79.31%->83.32% in its
profile window and85.79%->93.50% afterward. It is retained, not silently replaced
by low-overhead timing in any kernel analysis. The latest pair's fresh-process
controls give89.78%->95.54%; MBS1 retains a1.42-point recorded/control mismatch.

Full-profile windows:10-13; post:15-24. Latest a770 recorded:40-49; post:51-60.
All work allocations here are proportional. MBS increases apply to both replicas
and healthy; GA is unchanged, so local and global batches increase too.
Ladder pairs use EP8/8 healthy vs8/6 NEP,TP1. a8b uses EP16/16 vs16/12,TP2.
All use seq8192,DistOpt and Flex/HybridEP. a8b is GB300; others are GB200.
Each H/N pair shares an allocation. a500 MBS settings use different allocations;
the latest a770 settings and controls share one allocation.

## Mechanism hypotheses and relevance of the CPU changes

| Model | What the evidence supports | Could reducing NEP copy-path CPU overhead help? |
| --- | --- | --- |
| a315m | Already high post-profile parity with MBS4; no before/after tuning comparison. Full-profiler distortion is substantial. | Possibly, but limited clean residual headroom and no attributed CPU bottleneck. |
| a500m | Larger MBS improves parity in both windows. Reduced/full active-GEMM efficiency barely changes92.55%->93.42%; extra reduced GEMM-active time grows1.885->3.068ms. Larger work and changed overlap/submission scheduling are plausible, not an isolated causal attribution. | Plausible: same NEP copy path and short MBS1 iteration. No measured estimate of fix benefit. |
| a8b | Corrected original already93-94% parity. Historical profiles expose thousands of copy submissions, but those old placements and timing windows cannot explain the new result quantitatively. | Plausible fixed-cost reduction; no corrected-topology tuned or CPU-fix comparison yet. Not expected to eliminate bandwidth/capacity costs. |
| a770m | Latest gap falls56.08->31.12ms. Temporal improvement is chiefly first-forward parameter delivery and post-expert tail. More CPU calls finish before GPU ranges start at MBS2. This supports launch-headroom/amortization hypotheses but does not measure kernel starvation. | Most directly motivated candidate. Static code shows redundant singleton staging and repeated scalar copy setup. Benefit and absence of regression require A/B measurement. |

No CPU fix has yet produced a performance result in this table. More launch
headroom is not proof of GPU starvation. A CUDA end event submitted after the
CPU method returns can itself be delayed; operation ranges cannot distinguish
that from a missing kernel launch or dependency wait.

## Is reduced MoE efficiency a demonstrated primary cause?

Not for any topology-controlled full workload above. This is not a claim that
the efficiency loss is absent or that it cannot dominate another workload.

- a500's GPU-active expert efficiency deficit is real, but closing it selectively
  is not what the measured MBS comparison shows: the ratio improves only0.87pp.
- Older topology-controlled a770 profiles show89.07%->96.36% reduced/full expert
  efficiency, with the extra reduced active time falling4.065->1.974ms. Their
  same-window iteration gaps are145.350->144.825ms. That evidence does not show
  a large GEMM recovery causing a large absolute iteration-gap reduction.
- A separate exact-uniform EP16/12 shape experiment gives stronger evidence of
  intrinsic expert inefficiency:144experts, nine versus twelve experts/GPU,
  equal useful expert work. At H2688/F1856, base reduced/full active-GEMM
  efficiency is79.30%; doubling H and F gives88.96%. Independent H/F experiments
  leave wgrad near78% even while forward/dgrad recover. This is a regular-Adam,
  two-MoE-layer diagnostic, not the current full-workload DistOpt comparison.
  It establishes an expert-operation disadvantage, not that it dominates total
  iteration time. Kernel unions include possible execution contention, although
  forward has no recorded NCCL overlap and wgrad degradation repeats across
  markedly different overlap schedules.

Therefore CPU cleanup can remove submission/copy costs; it does not directly
fix shorter expert GEMM reduction dimensions or more gradient-output matrices.
Neither a claim that all poor parity is CPU overhead nor that a8b is primarily
GEMM-limited is supported by the current evidence.

## Historical coverage, not corrected acceptance results

These are the original ledger's post-warmup timing windows, usually NOT profiled
iterations. They are listed only to account for the rest of the workload set.
They must not be merged with the table above or used as causal improvement proof.

| Workload | Historical original parity | Historical tuned parity | Limitation |
| --- | ---: | ---: | --- |
| Ladder a1b/7b | 84.68% | Not established | Shared NVL domain across replicas |
| Ladder a2b/14b | 89.33% | Not established | Placement/provenance not fully validated |
| a3b/30b hybrid | 88.05% | 91.87% GB300;93.77% GB200 | Shared domain; hardware/allocations differ across tuning |
| a3b/30b transformer | 88.00% | 94.51% GB300;93.66% GB200 | Same limitations |
| Ladder a3b/30b | 99.33% | Not needed in that sample | Shared domain; not corrected RDMA result |
| Nemotron Nano | 94.84% | 92.04% replicated larger-batch GB300 | No established tuning improvement; shared domain |
| Nemotron Super | 103.82% | None | Noisy apparent speedup; incomplete provenance; substituted quantization recipe |
| a3b GDP latent/Muon | 94.81% | None | Inferred placement only; short timing sample |
| Nemotron Ultra | No valid NEP pair | Failed/OOM | Not a performance result |
| Dense8b and8b hybrid | N/A | N/A | Not MoE/NEP workloads |

Historical a8b tuning83.93%->96.12% with GA4->20 mixed cross-replica topology
and IB-leaf placement. Historical below-proportional a500 improvement did not
reproduce as a stable effect. Both are retired as clean failure/fix examples.
See NEP_WORKLOAD_CONFOUNDER_AUDIT.md and NEP_OWNER_PROGRESS_RESULTS.md.

## Delegated implementation

Agent Mill (full-context fork01a09141-a62e-7ac1-b6d6-45dc8ee1aedb) is implementing
the bounded changes on branch `nep/param-redistribution-cpu-20260911`, worktree
`/home/darfeen/Megatron-LM-nep-param-cpu-20260911`.
Original trusted checkout is not to be modified. Runtime metadata, not hardcoded
EP degrees, must drive every optimization. Collective order, prefetch position,
streams, bounded staging lifetime and optimizer/dispatcher behavior remain
unchanged. Implementation and validation completion are reported separately;
this document does not claim that the pending changes achieve a speedup.
