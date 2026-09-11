# NEP DistOpt Parameter Redistribution CPU Reduction

## Isolation

- Branch: `nep/param-redistribution-cpu-20260911`.
- Worktree: `/home/darfeen/Megatron-LM-nep-param-cpu-20260911`.
- Trusted runtime snapshot: `0fd1e023e`, including only the original two dirty files.
- Original production SHA256: `4676f164a95255bdcc7e2531a06ca92c41a2583bf6f9a7a85b3b74b64f26f2d1`.
- Original checkout/branch unchanged. Nothing pushed.
- Rollback: run the original trusted checkout or a worktree at the snapshot commit.

## Changes

Cache per-bundle static transfer group, source rank, proxy key, offsets, sizes,
and physical bucket references. Resolve proxy and model tensors on every call;
never retain cached tensor views or transfer storage. Singleton transfer groups
copy directly from owner proxy into the physical model tensor, retaining the
copy and tensor identity. Multi-participant pack/broadcast/unpack order is unchanged.

No changes to prefetch, streams, graph scopes, ordinary optimizer, gradient
synchronization, native DistOpt, Flex/HybridEP, or bounded transfer-buffer lifetime.
Batch-copy primitive changes deferred pending the isolated A/B measurement.

## Validation: 2026-09-11

Job 3007847, GB200 lyris0265, completed successfully in 1m34s.
Actual old/new methods exercised on CPU and CUDA with simulated broadcasts:
128 experts, EP8/4, EP8/6, EP16/12, EP32/28; full and reduced replicas, owners
and followers, heterogeneous local parameter shapes, two iterations with model
and proxy rebinding and transfer-buffer clearing.

- All output values identical; model storage identity and gradients preserved.
- Singleton scalar copy calls: 4,480 -> 2,240.
- Total scalar copy calls: 8,298 -> 6,058 (includes fake broadcast copy calls).
- All 182 broadcast submissions match in participants/source/order.
- `uv run --no-project --active isort` and Black run on changed Python code.
- This is not a real NCCL or full-model correctness test.
- Distributed EP8/6 MBS1/MBS2 same-allocation A/B subsequently completed below.

Logs: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/coreai_comparch_sysarch-nep.paramcopy-3007847.{out,err}`.

Job 3007873 exited before training: shared runner does not accept NEP-only case
order. The isolated diagnostic runner now permits NEP-only execution and analyzes
that case without requiring a healthy run. Production implementation unchanged.

## Distributed Performance Validation

Job 3007909 completed successfully in 16m35s on GB300 theia[0091-0094].
Same allocation/image, segment4, each replica entirely in nvlblk22, software
cliques 1000/1001 retaining cross-replica RDMA. Workload a770, H1280/F768,
128 experts/top-k6, 15 MoE layers, sequence8192, TP/CP/PP/ETP1, EP8/6,
GA3, MBS1 or MBS2 on both replicas, GBS42 or84. DistOpt, Flex/HybridEP,
nondivisible experts, overlap gradient reduction/parameter gather, unchanged
attention/router/Mamba graph scopes. No healthy comparison was rerun: this is
original NEP versus modified NEP, not a new healthy-relative parity estimate.

Order: original MBS1, modified MBS1, modified MBS2, original MBS2.
Each case completed60 iterations with zero skipped/NaN iterations and no errors.
All14 ranks captured targeted CUDA-event operation traces at iterations40-49.
All56 traces/560 measured rank-iterations passed the existing trace validator.

| Window | Original MBS1 ms | Modified MBS1 ms | Original MBS2 ms | Modified MBS2 ms |
|---|---:|---:|---:|---:|
| Pre29-38 | 464.98 | 440.52 | 723.23 | 705.46 |
| Measured40-49 | 470.15 | 442.58 | 718.54 | 709.20 |
| Post51-60 | 461.03 | 435.26 | 711.63 | 698.22 |

Measured improvement5.86% and1.30%; surrounding windows agree on direction.
Measured sample standard deviations:1.91/2.02ms at MBS1 and11.44/11.62ms
at MBS2. One allocation is not a universal no-regression guarantee; MBS2 gain
is small relative to per-iteration variability.

Mean full-owner(rank0-5) CUDA-event iteration spans:470.347->442.768ms and
718.705->709.358ms, agreeing with rank0 log timing.

| Redistribution CPU wall time, summed calls | Original MBS1 | Modified MBS1 | Original MBS2 | Modified MBS2 |
|---|---:|---:|---:|---:|
| Full owners0-5 | 19.382 | 17.630 | 16.464 | 16.264 |
| Reduced owners8-13 | 19.086 | 8.248 | 17.194 | 7.767 |
| Full followers6-7 | 15.070 | 11.368 | 12.592 | 10.302 |

Reduced-owner redistribution GPU operation ranges:14.007->3.736ms(MBS1),
4.025->1.746ms(MBS2). These are elapsed operation ranges, not kernel busy
time or exposed communication. Full-owner range increased14.715->16.307ms
at MBS1 despite lower iteration time; this must not be concealed as universal
per-operation improvement. Scheduling/dependency residency can move.

The largest observed host saving is on singleton reduced owners, consistent
with removing one of two copies per local parameter. MBS1 benefits more in
iteration time. This experiment does not establish exact GPU-starvation
milliseconds or attribute all end-to-end gains to one mechanism.

Exact scalar-copy output equality, gradient and storage preservation were
verified in the CPU/CUDA mapping test. Training losses matched initially but
diverged later between independent runs; full-training bitwise equivalence is
not established, and no claim that the divergence is proven harmless is made.
No larger-degree real distributed run or ordinary-optimizer rerun was performed.
Those code paths remain unchanged; mapping tests cover their ownership degrees,
not their full distributed execution. The scoped EP8 performance gate has no
observed regression, but broad feature/numerical validation is not complete.

Artifacts: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3007909/`.
Analysis: `param_transfer_analysis.json` under that directory (all rank rows and
all timing windows). Original runtime SHA remained unchanged throughout.
