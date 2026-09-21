# Remove Unintended Graph/Prefetch Launch Ordering

## Scoped Fix

For the validated native full-MoE graph benchmark, set
`NEP_LAUNCH_ORDER_IMPLICIT=0`. The benchmark launcher passes this through as
`NCCL_LAUNCH_ORDER_IMPLICIT=0`. The generic launcher still defaults to 1;
this is not a blanket change for untested workloads/topologies.

Runtime code is unchanged from b798119e9. Parameter AG -> redistribution ->
parameter-use device dependencies and per-process-group collective order
remain intact. Only the implicit cross-communicator launch chain is removed:
next-bucket redistribution must not hold up the current layer's TP launch
while it waits for its own parameter AG. No new streams, CPU polling, tensor
copies, bucketing changes, gradient scaling changes, or SM tuning were added.

Rollback: set `NEP_LAUNCH_ORDER_IMPLICIT=1`. The original worktree
`nep-static-slot-capacity-20260921` remains unchanged.

## Same-Allocation Results

Job3121210, GB200, eight nodes/segment4, one replica per NVLink domain:
nvlblk03 (lyris0044,0045,0047,0048) and nvlblk13 (0217,0223,0225,0226).
Original transport settings, including no PXN override. Healthy EP16/16;
NEP EP16/12. Depth-reduced a8b latent workload, 15 MoE layers, H4608,
latent1152, F3072, E512, top-k6, shared F6144, TP2/CP1/ETP1/PP1,
MBS2/GA8/sequence8192, healthy GBS256 and NEP GBS224 (proportional).
DistOpt, HybridEP, exact uniform capacity-padded routing, selective MoE
recompute, local CUDA graphs covering Mamba/attention/full MoE.

All cases ran40steps. Below are matched profiled steps25-27, averaging
full-replica owners0-11. Parity is healthy/NEP elapsed time, not throughput
normalized by the unequal total GPU counts. Ranges are elapsed GPU phase
ranges, not pure kernel service times.

| Metric (ms unless %) | Healthy order1 | NEP order1 | Healthy order0 | NEP order0 |
|---|---:|---:|---:|---:|
| First forward | 173.228 | 271.061 | 173.211 | 205.679 |
| Other forwards combined | 1116.151 | 1134.294 | 1116.698 | 1119.053 |
| Nonfinal backwards combined | 1658.082 | 1670.961 | 1660.355 | 1658.967 |
| Final backward | 246.024 | 293.803 | 246.032 | 276.237 |
| Outside forward/backward | 50.961 | 72.677 | 51.729 | 71.831 |
| Iteration | 3244.446 | 3442.795 | 3248.025 | 3331.767 |
| First-forward parity | - | 63.907% | - | 84.214% |
| Iteration parity | - | 94.239% | - | 97.487% |

NEP first forward decreased65.382ms (24.1%); its excess over the corresponding
healthy control decreased97.833 ->32.468ms. Not full first-forward parity.
The per-step owner means were270.643-271.305ms before and203.323-210.105ms
after. There were no measured backward/tail regressions in this comparison.

Rank0 Nsight graph-node evidence: all seven >1ms first-forward graph-internal
gaps per iteration disappear. Total graph-internal gaps53.996 ->2.043ms;
HybridEP-only graph residency51.099 ->33.548ms. These gaps are not device
idle: next-bucket prefetch was running during many of them. The removed
implicit ordering was preventing independent current-layer graph progress.

All32/28 ranks have Nsight SQLite/reports and CUDA-event records with native
AG/RS, NEP redistribution/Gather, process groups, payloads, host submission,
and GPU completion. Instrumentation checks passed. Logged post-profile
parity (steps30-40) was94.460% before and97.396% after; these are a distortion
cross-check only, not substitutes for the matched profiled measurements.

## Correctness Scope

Small job3121169: EP8/6, four MoE layers, MBS2/GA2/sequence1024, full graphs,
12steps per healthy/NEP ordering1/0 case; all completed without hangs, all-rank
profiles retained. Real-NCCL compact parameter and native AG/RS/prefetch
tests checked exact parameter values, nondivisible placement, followers,
buffer reuse, forced synchronization, and independent GPU progress.

Healthy small-run loss/grad norms matched exactly across settings. NEP was
not bitwise-identical: first2steps matched, then small differences appeared
(step3 loss relative4.5e-5, step12 relative0.00463). The long a8b runs also
diverged numerically after initially near-identical values. Exact training
trajectory equivalence and graph-vs-eager accuracy are NOT established by
these benchmarks. Do not present successful smoke/value tests as proving
either of those broader claims or arbitrary-degree deadlock freedom.

Initial small job3121140 failed in the original ordering1 NEP case with
mlx5dv_reg_dmabuf_mr error524, before reaching ordering0. The small retry
used NCCL_PXN_DISABLE=1 uniformly; the main a8b comparison did not.

## Artifacts / Reproduction

Artifacts:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3121210/`

- `launch_order_comparison.json`: all-rank/per-iteration disjoint phases.
- `phase_summary.txt`: owner phase summary.
- `launch_order{0,1}_{healthy,nep}/nsys/ep16/a8b_120b_latentmoe_1t/{healthy,nep}/3121210/`:
  logs, launch manifests, `cupti_profile/rank-*.json`, `nsys_profile/rank-*.sqlite`.
- NEP rank0 `rank-0.graph_analysis.json` in each Nsight directory: graph gaps,
  kernel partition, and neighboring collective launches.

Experiment harness in the main working checkout:
`scripts/nonuniform/run_a8b_graph_launch_order.sh` runs all four cases;
`run_a8b_graph_attribution.sh` defaults to the validated ordering0 worktree.
`analyze_graph_launch_order.py 3121210` regenerates the phase accounting.
The checkpoint b798119e9 plus launcher commit9212d8e9a identify the exact
runtime and launcher behavior tested.

## Saved Checkpoint (2026-09-21)

- Branch: `nep-graph-launch-order-20260921` on the personal fork
  `daiyaanarfeen/Megatron-LM`.
- Annotated checkpoint tag: `nep-graph-order-parity-20260921`.
- The runtime remains identical to the tested version; only this provenance
  record was added after validation. The tag records a performance result,
  not a claim of full training numerical equivalence.
- Exact harness snapshot (including the local profiling/analysis dependencies,
  launchers, storage rules, and journal as captured before this save):
  `saved_checkpoint/harness-snapshot-20260921.tar.gz` under the artifact root
  above. This is an archival snapshot, not a new production-code change.
- Archive SHA-256:
  `71a52045786db25d39f2fe01a60cd1333e03c6afcfb2f6f98171d8ba03767fb7`.
- Raw traces and benchmark data remain on Lustre, not in Git. No unrelated
  working-tree edits were included in the checkpoint commits.
