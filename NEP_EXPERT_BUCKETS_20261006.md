# Expert Bucket Sweep, 2026-10-06

Isolated from `nep-high-parity-delivery-20261005` at `7c23ac8e2`.
No training runtime changes. Revert by using that parent branch.

## Question

Does reducing the NEP expert bucket size shorten first-forward parameter
delivery and reduce final-backward interference with native compute/model-EP?
Test 3 (reference), 5, and 6 expert buckets using the existing environment
setting. Do not change dense buckets or native healthy bucketing.

## Fixed Controls

- Actual a8b latent hybrid/MTP saved memory-limited architecture, 15 MoE layers;
  H4608, latent1152, expert F3072, E512, top-k6.
- Healthy EP16/16, NEP EP16/12; TP2, CP1, ETP1, PP1.
- MBS2, GA8, sequence8192. Local BS128/128 healthy, 128/96 NEP.
- BF16 distributed optimizer, Flex/HybridEP, native MoE CUDA graphs.
- Exact randomized balanced routing, compact expert placement, ascending
  transfer order, one transfer stream, early delivery, prefetch depth1.
- Eight GB200 or GB300 nodes, four GPUs per node, segment4. A replica must fit
  in one NVL domain; clique isolation retains cross-replica RDMA transport.
- 35 iterations; measured25-30 after24 warmup. All-rank AG/RS, NEP redistribution,
  gradient Gather, phase, consumer readiness, and kernel traces.
- Nsight CUDA event tracing stays disabled to avoid its known false-dependency
  hang; custom GPU completion events remain enabled.

## Run Order

Each case has profiling and the previously authorized timing companion.
Repeat0: B3, H8K, B6, B5 (profile first).
Repeat1: B5, B6, H8K, B3 (companion first).
Sixteen cases, 15-minute case timeout, two-hour allocation cap.

The audit requires identical initialized weights, routing, dense buckets,
ordered expert slots/sizes, total expert AG/RS payloads, and destination-exact
redistribution bytes. Only expert bucket boundaries and expert collective
counts may differ. Numerical smoke tests run before the sweep.

## Acceptance

Report owner iteration parity against the fresh healthy control separately
for matched profiled iterations and companions. Report first forward, other
forward, nonfinal backward, final backward, and outside-F/B deltas. Check all
ranks, actual bucket sizes, memory, submission/completion, and native EP/TP
kernel durations. Overlapping residency differences are not additive causes.
Do not accept a first-forward gain that hides a backward/iteration regression.

## Preflight

Three CPU control-normalization tests pass. Python AST, shell syntax, git
whitespace, and unchanged `megatron/` runtime checks pass. Import order fixed
with `uv run isort`; formatting applied to modified Python files.
