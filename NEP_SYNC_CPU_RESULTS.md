# Incremental NEP Sync CPU Cleanup

Date: 2026-09-14. Branch: nep/sync-cpu-20260914.
Trusted runtime is unchanged. No changes to native DDP, optimizer, dispatcher,
stream ordering, prefetch, batch sizes, or temporary-buffer lifetime.

## Scope and Acceptance

Remove redundant copies and repeated metadata work; use native PyTorch batched
copies rather than introducing custom kernels. Validate each step before moving
on. Do not infer that every residual cost is unavoidable merely because these
optimizations pass. Metadata/list construction, native runtime and dependency
scheduling still have implementation costs.

All benchmarks: a770, EP8/8 healthy or EP8/6 NEP, E128 top-k6, H1280/F768,
seq8192, GA3, DistOpt + Flex/HybridEP, same existing graph scopes. MBS1 or2
on both replicas; proportional replica batch allocation. Four GB200 nodes,
replicas contained within NVL domains, software cliques forcing cross-replica
RDMA. Every benchmark has targeted CPU/CUDA-event profiles on all ranks,
iterations40-49 of60. No profiler-overhead subtraction or mixing windows.

Timing below is rank0 logged measured-window mean, matching full-owner mean
CUDA-event iteration ranges to roughly0.2ms. The analyzer also retains per-step
means across full-owner ranks0-5 and their sample SD.

## Gate 1: Metadata Cache and Singleton Direct Copy

Inherited f1354ff70, trusted reference0fd1e023e. Harness7060ca275.

- Job3054627 failed before training because a duplicate Pyxis mount argument
  replaced the code mounts. Fixed launcher only; no NEP execution occurred.
- Job3054658 completed18m17s on lyris[0095-0096,0099-0100], segment4.
- Correctness-only live shadow execution:224 parameter transfers and192 owner
  gradient staging calls matched trusted methods exactly; destination storage
  identity preserved. Owners and followers participate in real NCCL transfers.
  These checks precede, and are disabled during, performance cases.
- MBS1 original458.69 versus cached451.29ms (1.61% faster).
- MBS2 original724.01 versus cached732.02ms initially (1.11% slower). Held
  acceptance and ran a reversed-order repeat instead of ignoring the regression.
- Job3054733 completed16m03s on the same four nodes: original/cached/cached/
  original MBS2 measured733.37/728.64/723.41/739.72ms. No repeatable MBS2
  regression established; cached average726.03 versus original736.55ms.
- Original run's extra MBS2 time was mostly ordinary later-forward regions;
  that location alone was not used to dismiss the first negative result.

## Gate 2: Batched Parameter Copies

Commit2fe05a928. Replace the three scalar-copy loops (singleton, pack, unpack)
with torch._foreach_copy_, resolving live tensors at each call. Runtime diff
from cached version:13 additions/3 deletions, one method in nonuniform_ep.py.

- Job3054823 completed18m18s on the same four GB200 nodes.
- Actual methods pass CPU/CUDA FP32/BF16 mapping tests at EP8/4,8/6,16/12,
  32/28, including rebinding model and proxy tensors, gradients/storage identity,
  and identical simulated broadcast participants/order.
- Real EP8/6 shadow checks again pass224 parameter/192 gradient comparisons.
- Cached versus parameter-batched measured means:
  - MBS1:450.10 ->441.84ms (1.84% faster).
  - MBS2:724.34 ->716.86ms (1.03% faster).
- Full-owner parameter redistribution CPU wall time17.528->8.364ms at MBS1,
  15.483->8.100ms at MBS2. Reduced-owner8.666->2.779ms and7.882->2.937ms.
- Full-owner redistribution GPU operation-range union14.954->9.275ms at MBS1,
  11.531->9.485ms at MBS2. These include dependencies/host spacing; not pure
  transfer time or exclusive communication residency.
- Existing native gradient staging remains unchanged in this gate.

## Gate 3: Gradient Staging (In Progress)

Job3054926, same nodes/segment. Candidate adds cached integer/key-only gradient
copy descriptors and one foreach copy per existing context. Preserves copy-done
events and native reduce-scatter ordering. Reference is gate2 commit2fe05a928.

Additional tests compare actual old/new staging with padding, heterogeneous slot
shapes, source/destination guards and rebound grad/source storage, across all
four EP pairs and FP32/BF16 CPU/CUDA. Then live shadow and MBS1/2 A/B.
Do not consider this gate accepted before its results are analyzed.

## Artifacts and Limits

Lustre root: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/
soft_clique_baseline/JOB_ID/. Each analyzed job has sync_cpu_analysis.json.
Analyzer: /home/darfeen/Megatron-LM/scripts/nonuniform/analyze_nep_sync_cpu.py.

Full-training loss trajectories differ across independent processes. Exact live
operation replay addresses the changed copy transformations, not a general proof
of bitwise full-training determinism. No larger-degree full-model performance
validation or new healthy-relative parity result is claimed at this stage.
