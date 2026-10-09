# Hard MBS1/GA2 graph retest, 2026-09-21

Job3121678 completed0:0 on GB200 in11:29. Both cases completed45 iterations;
measured25-34 follow24 warmup iterations. No runtime changes for this test.

## Setup

Same depth-reduced a8b latent workload as3121210:15 MoE layers with native
Mamba/attention/MoE pattern; H4608, latent1152, expertF3072, E512, topk6,
sharedF6144, seq8192. MBS1/GA2, TP2/CP1/ETP1/PP1. HealthyEP16/16 uses
local16/16, GBS32; NEPEP16/12 uses local16/12, GBS28, proportional.
DistOpt, Flex/HybridEP, exact-uniform capacity-padded routing, selective
MoE recompute, native local Mamba/attention/full-MoE graphs. Saved runtime
8abb29fca, implicit launch ordering0. This changes only MBS/GA from the
saved MBS2/GA8 model; it is not an exact historical low-GA rerun.

Same eight-node allocation, segment4: lyris0170-0173 and0174,0176-0178
(last node unused for NEP). Both replicas physically in nvlblk10, with
distinct MNNVL software cliques1000/1001 enforcing cross-replica RDMA.
All-rank logs show GDRDMA and cliqueSize1. Each replica is physically
contained within one NVLink domain; these are NOT two physical domains.

## Matched owner results

Per-iteration means over ranks0-11, then averaged over steps25-34.
Parity = healthy/NEP. Elapsed schedule phases, not pure kernel service times.

| Phase | Healthy ms | NEP ms | Extra ms | Parity |
|---|---:|---:|---:|---:|
| First forward | 98.762 | 137.665 | 38.903 | 71.74% |
| Other forward | 82.712 | 83.119 | 0.407 | 99.51% |
| Nonfinal backward | 128.213 | 128.587 | 0.374 | 99.71% |
| Final backward | 141.793 | 171.935 | 30.142 | 82.47% |
| Outside forward/backward | 60.083 | 87.241 | 27.158 | 68.87% |
| Entire iteration | 511.562 | 608.547 | 96.985 | 84.06% |

Iteration slowdown18.96%. Standard deviations of per-iteration owner means:
healthy1.596ms, NEP3.027ms. Ranges509.244-513.932 and604.578-613.854ms.
Outside-phase time includes startup, microbatch gaps, gradient-finalization/
optimizer boundaries and remaining tail; it is not one identified operation.
The phase gaps are disjoint and sum to the total iteration gap.

NEP reduced means: firstF132.351, otherF83.191, nonfinalB129.747,
finalB172.254, outside91.007, iteration608.549ms. Full-replica follower
iteration608.543ms. No much-slower follower iteration is hidden here.
99.2% of the gap is in firstF, finalB and outside-phase time; ordinary
forward/backward remain near parity. This is not operation-level attribution.

## Profiling checks

All32 healthy and28NEP profiles validated: nativeAG/RS, group/payload metadata,
host submission and GPU completion, NEP redistribution on all ranks and
gradient staging on owners. Every measured iteration has two complete
forward/backward microbatches and30 MoE graph replays per direction.
Nsight graph-node SQLite files present on every rank. All45steps finite and
non-skipped; no hang/OOM. Full graph-vs-eager numerical equivalence is not
established; saved-checkpoint correctness qualifications still apply.

Separate training-log windows, NOT substituted for measured phase timings:

| Window | Healthy ms | NEP ms | Parity |
|---|---:|---:|---:|
| Pre-profile15-23 | 496.567 | 592.522 | 83.80% |
| Measured25-34 | 511.180 | 608.170 | 84.05% |
| Post-profile36-45 | 497.850 | 592.960 | 83.96% |

Low parity persists outside profiling; measured/post parity differs by0.1pp.
This is a distortion/drift check, not exact profiler-cost subtraction.

## Artifacts

Root: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3121678/`.
`hard_graph_phase_analysis.json` retains per-rank/per-iteration records,
cohort statistics and comparisons. Profiles are under
`hard_mbs1_ga2_{healthy,nep}/nsys/ep16/a8b_120b_latentmoe_1t/{healthy,nep}/3121678/`.

Harness: `scripts/nonuniform/run_nep_hard_graph_ab.sh` and
`experiments/nep_hard_graph_20260921/nep_param_transfer_soft_clique.sh`.
Reanalyze: `python3 scripts/nonuniform/analyze_nep_hard_graph.py 3121678`.

## Phase-cause investigation

Existing profiles only; no new training jobs or production code edits.
All-rank operation records cover steps25-34. Kernel inspection covers ranks
0,8,11,15,16 in both cases, with targeted rank2/3/5 dependency inspection.
Do not add kernel residency differences to the phase gaps: kernels overlap,
and one rank's wait can reflect another rank's delay.

### First forward: +38.903ms

Parameter delivery creates repeated rank skew before model-EP dispatch.
All-gather is followed by post-AG packing, broadcasts and unpacking. On each
follower these operations share a transfer stream. Some broadcast groups span
adjacent followers, creating a dependency chain through followers12-15.
Owners0-11 cannot finish their broadcasts independently of that chain.

Concrete step28, second parameter bucket, times relative to rank0 firstF:

| Broadcast source | Participants | GPU ready ms | Completion ms |
|---|---|---:|---:|
| 0 | 0,12 | 18.823 | 19.489 |
| 2 | 2,12,13 | 19.461 | 20.589 |
| 5 | 5,13,14 | 20.081 | 22.215 |
| 8 | 8,14,15 | 20.686 | 23.788 |
| 11 | 11,15 | 21.616 | 25.194 |

Follower15 cannot issue its source11 transfer on the GPU until its preceding
transfers complete; its source11 ready/completion is24.910/25.185ms.
Across10 iterations, owner11 parameter-redistribution completion trails owner0
by2.887,5.722,8.179,6.177,7.054,7.080,6.826,7.469ms across the8 buckets.
Those values are skew, NOT additive iteration costs.

Early owners enter the MoE layer and wait for later participants. In step28,
forward graph ordinal3 spans15.745-23.773ms on rank0 versus
21.456-23.765ms on rank11, relative to rank0's first graph. Rank0's firstF
HybridEP ag_nvl kernel residency rises1.977 ->30.934ms, and device_sync
0.479 ->3.531ms; MoE graph ranges rise33.914 ->70.710ms. This is mostly
waiting within dispatch ranges, NOT slower expert GEMMs or extra token work.
GPU-idle time in rank0 firstF is1.246 ->1.481ms, inconsistent with CPU
starvation as the dominant explanation. There are also parameter-copy costs.

Underlying factors: logical owners carry more parameters; firstF AG output
per rank0 is6.795GB healthy versus9.130GB NEP (+34.375%). Native healthy has
5 expert AG/RS buckets versus8 NEP, with3 dense buckets in both. More, smaller
expert buckets change prefetch coverage. This is the actual implementation
comparison, not an iso-expert-bucket causal experiment. The records do not
uniquely split all38.9ms into increased AG volume versus broadcast serialization.

Code: NonuniformEPNCCLParamAndGradBucketGroup.start_param_sync,
_scatter_nep_distopt_owner_params_to_physical_holders and finish_param_sync
in the pinned worktree's megatron/core/distributed/nonuniform_ep.py.

### Final backward: +30.142ms

Gradient Gather, staging copies and owner RS run during backward. Native
backward kernels incur GPU-side gaps, followed by TP and HybridEP peer waits.
This is not an unhidden CPU expert-submission problem: all owner finalB graph
host calls complete at least87.294ms before their GPU operation ranges start.
There is no comparable ordinary-backward slowdown.

Concrete step28, times relative to rank0 finalB start (rank2 Nsight aligned
to the CUDA-event trace):

1. Rank2 native graph's vectorized kernel completes35.940ms; its following
   1632-byte memset completes35.943ms.
2. The next RMSNorm backward kernel does not start until41.649ms: a5.706ms
   gap inside an already-submitted graph. NEP Gather SendRecv is resident
   34.852-41.892ms on another stream.
3. TP partner rank3 waits in model-TP AG36.272-41.949ms.
4. Rank0 subsequently waits in HybridEP device_sync45.374-49.685ms as
   delayed model-EP participants catch up.

This RMSNorm gap in backward graph ordinal2 occurs in all10 NEP final
backwards (1.426-5.710ms); healthy finalB and both ordinary backwards have
about0.003ms at that position. Across all RMSNorm positions on rank2,
finalB gaps sum8.058ms healthy versus15.692ms NEP. Healthy also experiences
some such stalls while its own RS runs; this is not uniquely a NEP kernel.
Rank0 finalB device_sync residency increases3.365 ->20.931ms. These are
different views of propagated delays, not amounts to sum together.

Reduced-replica lateness is not the main RS bottleneck in this run. Across
960 matched owner-pair/bucket/iteration records, full-replica owners reach
RS input readiness later in92.08% of cases (50us tolerance), averaging
4.753ms later. They have extra intra-replica Gather/accumulation work.
The reduced replica is later in7.29% of cases.

Confirmed: GPU-side native-work delay during overlapping synchronization,
not CPU starvation, not a uniform ordinary-compute slowdown, and usually
not late reduced-replica RS readiness. Strong candidate: resource/scheduling
interference from resident NEP NCCL and gradient-buffer work. The trace does
NOT expose complete CUDA graph dependency edges or SM admission decisions,
so it cannot conclusively distinguish resource contention from an unrecorded
graph dependency, or assign separate causal milliseconds to Gather versus RS.
Do not claim this lowest-level mechanism is definitively solved.

### Outside forward/backward: +27.158ms

Owner-mean elapsed phase accounting:

| Component | Healthy ms | NEP ms | Delta ms |
|---|---:|---:|---:|
| Before firstF | 3.106 | 5.490 | +2.384 |
| Backward to optimizer | 33.462 | 49.118 | +15.656 |
| Optimizer | 18.619 | 27.654 | +9.035 |
| After optimizer | 4.875 | 4.959 | +0.084 |

The backward-to-optimizer increase is tightly accounted for:

- Final ddp.finish_grad_sync:16.702 ->16.704ms, essentially unchanged.
- Gradient scaling:14.054 ->25.901ms (+11.847ms). NEP calls native physical
  buffer scaling, then separately scales persistent logical-owner buffers.
  The owner-only extra pass is11.428ms; physical scaling rises0.419ms.
- Post-backward TP gradient reductions:1.535 ->2.979ms (+1.444ms).
- Post-backward dense-group scalar/token-count reductions:0.346 ->2.707ms
  (+2.361ms). These are elapsed collective ranges, including participant wait,
  not additional bandwidth-service time. With scaling and grad-sync these
  explain15.654ms of the15.656ms increase.

Optimizer increase:

- Grad norm:2.310 ->6.018ms (+3.708ms). Its scalar reduction's mean
  pre-last-participant wait is0.352 ->3.426ms. The latest NEP participants
  are reduced-replica ranks16/18/19/23, after their larger physical-buffer
  scaling work. After-last-ready elapsed is0.032 ->0.057ms, not3ms of wire time.
- Adam/update scope:13.252 ->17.588ms (+4.336ms), including parameter copies.
  About33% more logical expert optimizer payload per owner is consistent with
  this32.7% increase. Nested parameter-copy time2.902 ->3.843ms must NOT be
  added again to the update scope.
- Remaining optimizer work, including clipping and unscoped boundaries,
  contributes about0.99ms additional time.

Startup's extra2.38ms is extra gradient-buffer clearing: rank0 vectorized
  zeroing kernel residency2.857 ->5.226ms. The NEP zero_grad_buffer override
  resets both native physical and logical-owner buffers.

Code: NonuniformEPDistributedDataParallel.zero_grad_buffer/scale_gradients;
native finalize_model_grads per-token normalization; native chained optimizer
grad-norm and update paths. Extra buffer passes are implementation-specific
costs to audit, not automatically unavoidable NEP communication costs.

### Reproducible evidence

Lustre job directory `phase_causes/` contains all-rank matched event records,
rank kernel partitions, `phase_scope_and_readiness_summary.json`,
`rs_participant_readiness.json` and `rank2_rmsnorm_launch_gaps.json`.
`scripts/nonuniform/analyze_hard_phase_causes.py` reproduces per-rank graph,
kernel, communication and optimizer-scope accounting. GPU-idle means neither
kernels nor recorded memcpy/memset are active on the rank, not merely no GEMM.
Kernel categories can overlap; the partition preserves that overlap explicitly.
