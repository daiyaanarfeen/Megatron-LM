# Controlled NEP Placement and Parameter Scheduling

Approved September 29, 2026. Base: e0113b468 (exact destination payloads).
Implementation branch: nep-placement-controlled-20260929. Trusted branch unchanged.

## Policies

| Case | Placement | Parameter owner order | Active lanes |
|---|---|---|---:|
| A | compact | ascending | 1 |
| B | compact | disjoint-first | 1 |
| C | compact | ascending | 2 |
| D | compact | disjoint-first | 2 |
| E | spread | ascending | 1 |
| F | spread | ascending | 2 |

Spread keeps owner-local expert IDs, per-rank expert counts, per-owner send totals
and per-follower receive totals unchanged. Only pair membership/pair payloads change.
For the target EP16/12 layout every spread owner connects to every follower, so
disjoint-first reduces to ascending. No redundant spread/order cases are needed.

## Implementation Controls

- NEP_ABLATION_PLACEMENT=compact|spread, default compact.
- NEP_ABLATION_PARAM_ORDER=waves|ascending, default waves.
- NEP_ABLATION_PARAM_LANES=0|1|2. Zero retains existing resource behavior. One/two
  use the same transfer implementation, two persistent preallocated buffer lanes,
  two streams, and matching readiness/completion edges. Owner ID modulo active
  lane count selects the lane. No new communicator or progress thread is added.
- NEP_ABLATION_FIXED_GRAD_ORDER=1 uses ascending backward owner batches in every
  ablation cell. It is off for the initial old-versus-new one-lane regression gate,
  to avoid confusing that comparison with a changed gradient scheduling policy.
- The existing AG-ready dependency fans out to the lanes. The original parameter
  completion event follows both lanes; no consumer may use undelivered parameters.
- Existing native Megatron optimizer/dispatcher files remain untouched.

## Gates Before EP16

1. Unit and real-NCCL fresh-parameter tests for all six policies, nondivisible
   layouts, overlap on/off, force-sync, repeated buffer reuse and native AG/RS.
2. EP8/6 regression: reference/one-lane/one-lane/reference order, plus compact
   and spread two-lane training execution checks. Profile every rank. The spread
   execution check is NOT placement performance evidence until initialization,
   routing and bucket manifests are controlled and validated.
3. User-approved no-profiler companion reference/one-lane/one-lane/reference
   runs. Match iterations15-20; do not substitute post-profile timing.
4. Show scoped diff, exact payload manifests, correctness and regression evidence
   to the user before starting the EP16/12 sweep.

Initial EP8 workload: four MoE layers, H4608/latent1152/F3072/E512/top-k6,
MBS2/GA2/S1024, TP2/CP1/ETP1, DistOpt, Flex/HybridEP, full MoE CUDA graphs.
Replica batches16/12; healthy16/16 if used. Explicit segment2 and software cliques
restrict cross-replica collectives to RDMA even inside one physical NVLink domain.

## Acceptance Requirements For Placement Results

- Identical initial logical parameters and optimizer state by global expert ID;
  merely using identical seeds is insufficient. No checkpoint implementation changes.
- Identical inputs and exact balanced per-expert/per-peer routing counts; identical
  GEMM shapes, graph coverage, real bucket boundaries and native AG/RS payloads.
- Record every pair's required/actual payload and validate exact parameter values.
- Equal preallocated resources across lane modes; verify real GPU concurrency,
  not just multiple stream IDs or overlapped host enqueue intervals.
- Same GPU type/topology policy; verify transport and placement. Repeat in
  counterbalanced order; retain matched measured iteration indices and variability.
- Redistribution-only all-ready replay separates transfer mechanics from AG
  readiness and compute. Full-training placement results include native token
  dispatch and backward-placement effects; do not label them parameter-only effects.
- Native collectives must not change across scheduling-only comparisons.
- Report full-owner iteration parity first. Separate phase times, payloads,
  readiness waits, collective elapsed ranges and kernel overlap. Do not sum
  overlapping intervals into an additive causal overhead claim.

## Status

September30 final continuation: the six-policy EP16/12 study ran with
counterbalanced repeats. Jobs3200495 (profiled replay),3200606 (profiled training
and matched timing companions),3200857 (replay timing companions and anchors)
all completed0:0. All scoped correctness/control checks passed. Primary measured
profiler-off training means A/B/C/D/E/F are83.850/83.617/83.550/83.567/90.075/
86.842ms; corresponding all-ready delivery replay means are5.636/5.277/4.247/
4.479/12.423/7.548ms. These are distinct timing experiments, not pure wire times.

STOPPED further attribution on a newly observed host-stall confounder in the last
profiled B replay anchor (6.013->9.306ms), localized to CPU submission on ranks9/10.
Their non-thread-CPU elapsed time increases by about3.25/2.75ms; CUDA runtime calls
also lengthen. OS scheduling versus driver blocking is unresolved in these traces.
The profiler-off B repeats remain5.272/5.282ms, and no samples were discarded.
Do not use the anomalous anchor for a constant profiler correction or claim fully
resolved cross-protocol causality. No more jobs submitted; user notified per the
stop-on-confounder requirement. Full measurements and diagnosis are in the journal.

September30 continuation: the user approved randomized exact routing and separate
parameter-only/end-to-end measurements. The routing stop below is resolved for
the EP8/6 gate: job3199910 passed CPU/CUDA/graph checks, and job3199948 completed
both profiled compact/spread cases. All4096 logical expert tensors, per-rank dense
weights, actual expert/dense bucket layouts, routing maps, native collective
manifests and aggregate parameter bytes match. Small unique-token/peer sampling
differences are measured, not forced equal or interpreted as measured wire bytes.
The current seeded edge-swap construction removes adjacent-ID co-selection while
preserving exact expert degrees and each token's distinct top-k choices.

Gate owner iteration times:153.944ms compact and158.759ms spread (seed101 only,
one run order, not a final ranking). All-ready six-policy replay3200181 completed;
it profiles real initialized owner layouts before optimizer construction and
excludes AG/RS, compute and token dispatch by design. All six policies preserve
exact values and7,247,757,312 aggregate source/receiver bytes per repetition.
Mean maximum full-replica rank-local elapsed times in milliseconds:
A10.021, B9.250, C8.638, D8.610, E11.599, F9.919. These include launch/staging/wait
and are not pure transfer service or end-to-end training results. Two-lane GPU
kernel concurrency is verified; no concurrency occurs in single-lane policies.
EP16/12 compact/spread validation3200350 completed0:0 in9m20s on eight GB200
nodes, segment4, same per-microbatch model settings. All28 ranks passed the
weight/routing/bucket/native-collective/payload controls. Profiled owner means
were90.868/102.686ms; these single-order gate observations are not final ranking
evidence. Separate post-profile log means86.975/92.825ms are not substituted
into trace analysis or treated as causal profiler corrections. No six-policy
EP16 sweep submitted yet. Final ranking requires counterbalanced repeats and
matched-window profiling-overhead companions. The routing blocker is resolved.
This continuation changes only benchmark harness/tests, not NEP production code,
the native dispatcher or the optimizer. Detailed records are in the journal.

September30: STOPPED at CPU routing preflight per user instruction. No new GPU
job or implementation change. Exact-uniform per-expert balance does not preserve
unique token-to-destination counts under compact/spread. For1024 source tokens,
E512/top-k6, token-GPU pairs are1144/1432 atEP8/6 and1272/2096 atEP16/12.
Per-rank expert assignments remain identical (768 and384, respectively).
These are routing-map counts, not measured transfer bytes or runtime costs.
The full-training placement comparison therefore fails its routing-invariance
gate pending a control/interpretation decision. All-ready parameter-only replay
can isolate transfer scheduling without this native-token-routing difference.
See September30 journal entry for per-follower counts and method.

Job3190428 completed0:0 on GB200 lyris0059 in6m14s:98 unit tests, all six
real-NCCL policy cells and native AG/RS prefetch under one/two lanes passed.
Every policy's trace contains663,552 aggregate source/receiver bytes with exact
pairwise split matching. These are small synthetic test payloads, not bandwidth
measurements. Lane-stream counts must be compared within an exercise, because
the two tested layouts create different stream objects.

Saved implementation/tests as8cfd198d2. EP8/6 gate job3190514 is running on four
GB200 nodes lyris0217/0218/0258/0260, segment2. Runs are compact reference/one/
one/reference (profiled), compact two-lane and spread two-lane execution checks,
then compact reference/one/one/reference (profiling disabled by user approval).
Profiler-off entrypoint strips only profiler CLI options and installs no profiling
wrappers; model, graph, batch and optimizer options remain unchanged.
No EP16 sweep has been submitted. The six-policy study and profiling-off
healthy/current/best-candidate confirmations remain gated on the checks above.

## EP8/6 Gate Results

Job3190514 completed0:0 in23m33s on September29. All ten cases completed25
iterations with zero skipped/NaN iterations. Replica0 nodes0217/0218 were in
nvlblk13; replica1 nodes0258/0260 were in nvlblk15. No collective timeout or OOM.
Both replicas used MBS2/GA2, with proportional replica batches16/12.

Matched steps15-20 after warmup; two runs per implementation, ordered R/C/C/R:

| Measurement | Destination-specific reference | Controlled one lane | Change |
|---|---:|---:|---:|
| Profiled full-owner iteration, ms | 151.743 +/-0.627 | 151.735 +/-0.626 | -0.005% |
| Profiling-disabled logged iteration, ms | 145.392 +/-0.300 | 145.217 +/-0.252 | -0.120% |

SD is across12 step observations (profiled observations average owner ranks0-5).
Profiler-off timing is the native rank0 log, not an all-rank phase measurement.
Do not substitute it into trace decompositions. Profiled native log means were
151.508/151.517ms: instrumentation elevation relative to the separate off runs
was4.21%/4.34%, respectively. These are not causal per-operation corrections.
Total-iteration regression gate passes within observed variation, not a speedup
claim or proof of universal no-regression. Profiled first-forward changed
38.481->38.765ms (+0.284); final backward16.369->16.518ms (+0.149);
outside F/B84.529->84.091ms (-0.438). Other F/B changed by less than0.006ms.

The compact two-lane execution check was153.170ms on full owners,0.95% slower
than pooled one-lane. First-forward was40.306ms. This single execution check
is not a final policy ranking. Actual transfer kernels on follower ranks6/7
overlap each other for29.322/30.053ms per first forward (zero in one-lane).
Those intervals include participant waiting: they establish kernel concurrency,
not simultaneous useful bandwidth or pure transfer service time.
Spread two-lane completed and passed payload checks, but its155.158ms timing
is NOT accepted placement-performance evidence: logical initialization and
backward policy differ from the compact gate. Do not plot it as a comparison.

All six profiled cases transfer7,247,757,312 aggregate parameter bytes per step,
with matching input/output splits for every participant pair and no parameter
broadcast. Compact reference/one/two non-parameter collective manifests match
exactly, including native AG/RS and gradient Gather. All14 ranks have native
AG/RS, parameter redistribution host/GPU records and Nsight SQLite/reports.
All recorded workload options agree after removing only profiler options and
TensorBoard output paths; effective off-mode options were checked on all ranks.

Resource tradeoff: controlled one/two modes retain identical two-lane buffers.
On logged owner ranks0/1 at step20, allocated memory is58,648.04MiB versus
reference51,682.04MiB (+6,966MiB,6.803GiB); max allocated58,846.13 versus
55,174.12MiB. One/two/spread-two memory observations are identical. This is an
opt-in experiment-resource cost, not a memory-neutral production replacement.

Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3190514
- controlled_gate_analysis.json and analyze_nep_controlled_gate.py
- controlled_gate_concurrency.json and analyze_nep_controlled_concurrency.py
- run_nep_cleanup_ab.sh (exact shared harness snapshot)

Implementation remains clean at8cfd198d2; trusted referencee0113b468 untouched.
Runtime changes remain confined to the two NEP modules. No EP16 job submitted.
Before the full study: implement/verify global-expert identity initialization,
routing/GEMM/bucket manifests and all-ready redistribution replay, then present
the controlled configuration before proceeding beyond this user review gate.
