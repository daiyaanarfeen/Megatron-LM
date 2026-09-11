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
- Distributed EP8/6 MBS1/MBS2 same-allocation A/B with all-rank CUDA-event
  profiling is pending. No claim of performance non-regression yet.

Logs: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/lyris/coreai_comparch_sysarch-nep.paramcopy-3007847.{out,err}`.

Job 3007873 exited before training: shared runner does not accept NEP-only case
order. The isolated diagnostic runner now permits NEP-only execution and analyzes
that case without requiring a healthy run. Production implementation unchanged.
