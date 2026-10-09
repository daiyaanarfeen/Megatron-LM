# Sequence length and reduced-replica lateness

## Controlled comparison

Job3073889, COMPLETED0:0,2026-09-16 11:16:33-11:34:56PDT,18:23.
Same GB200lyris[0253-0256],segment4, physical nvlblk15. Each replica remains
within that domain; software cliques1000/1001 enforce cross-replica RDMA,
preserving intra-replica NVLink. Named NeMo26.06 container unchanged.

Four sequential cases: S2048 healthy/NEP, S8192 healthy/NEP,60iterations each.
All-rank expert CPU/CUDA-event and AG/RS operation profiles40-49; timings
below use exactly those ten iterations, not post-profile replacement timings.
All cases completed without skipped/NaN iterations. Profiles16/14ranks each.

Only sequence length changes. H1280, expert F768, shared F1536,512experts,
top-k6,15MoE/15Mamba/3GQA; MBS2/2,GA3/3; healthyEP8/8 GBS96 versus NEP
EP8/6 GBS84, proportional batches48/48 versus48/36; TP/CP/PP/ETP1,
DistOpt/Flex/HybridEP/local graphs retained. Expected expert tokens per
microbatch rise384/288 to1536/1152 (full/reduced); token-dependent routing
and communication change naturally with sequence length. This is not an
isolated GEMM-only perturbation.

Reference93d313df4d55 and candidate51fdbbb11 remain clean and unchanged.
Analyzer asserts normalized arguments differ only in sequence length/output
paths; implementation hashes, bucket layouts, expert collective counts and
input/output bytes are identical across sequence points. Healthy retains its
native7expert buckets versus8logical NEP groups at both points.

## End-to-end results

Mean +/- sample SD across ten iteration means over full-owner ranks0-5:

| Sequence | Healthy ms | NEP ms | Owner parity | Extra ms |
|---|---:|---:|---:|---:|
| 2048 | 703.084 +/-15.514 | 861.614 +/-7.164 | 81.601% | 158.530 |
| 8192 | 865.972 +/-4.152 | 1025.773 +/-5.313 | 84.421% | 159.801 |

Separately, logged post-profile51-60 parity81.819% and84.570%; measured
logged parity81.592% and84.421%. No profiling-overhead subtraction. This
allocation's short-sequence parity is better than the prior allocation; use
the same-allocation control above, not old73% as the before value.

## Ordinary backward host-work exposure is reduced

Second microbatch, means over six owners and ten iterations, milliseconds:

| Measurement | S2048 full/reduced | S8192 full/reduced |
|---|---:|---:|
| Ordinary backward elapsed span | 95.383 /114.136 | 108.550 /110.644 |
| Expert backward active CPU time | 61.172 /76.913 | 62.471 /78.778 |
| Calls submitted fully before GPU range starts | 69.7% /43.4% | 100% /100% |
| Mean GPU-start minus CPU-start | 5.233 /3.221 | 42.332 /32.231 |
| Ordinary forward elapsed span | 85.265 /92.351 | 124.590 /131.046 |
| Expert forward active CPU time | 32.208 /41.014 | 32.268 /40.523 |

Backward reduced-minus-full gap falls18.753 ->2.094ms (88.8% reduction),
while additional expert CPU work does NOT fall:15.741 ->16.307ms. The larger
compute/dependency lead now puts every expert backward host call ahead of its
GPU event range. This supports hiding the ordinary backward submission cost,
not eliminating that cost or proving a particular change in GEMM FLOP/s.

Forward remains host-paced in the recorded expert calls: launch-ahead fraction
is0 for both cohorts at both sequence points. Its reduced-minus-full span
gap changes only7.086 ->6.456ms. More sequence compute does not automatically
create an expert-forward submission queue or hide all forward CPU work.

## Final backward and synchronization

Full/reduced paired markers, means over the same60samples:

| Measurement | S2048 ms | S8192 ms |
|---|---:|---:|
| Reduced final-B entry lag | 41.458 | 5.881 |
| Reduced final-B exit lag | 54.616 | 22.719 |
| Reduced final-B duration minus full | 13.158 | 16.838 |
| Full local final Gather/staging done after last expert B | 28.321 | 3.958 |
| Final RS wait after local ready until reduced ready | 30.400 | 22.216 |
| Final RS completion after both ready | 4.475 | 4.276 |
| Owner last expert B to optimizer start | 82.901 | 50.621 |

Most accumulated ordinary-phase entry skew disappears, but the reduced final
backward remains16.8ms longer. It therefore still arrives late for final RS.
These event ranges do not resolve that residual into GEMMs, extra gradient
hooks/staging submission, or HybridEP waits. Do not transfer the demonstrated
ordinary-backward improvement into a claim that final backward is solved.

Owner summed Gather input-ready-to-completion ranges fall99.659 ->14.674ms;
expert RS377.073 ->146.509ms. Payloads are unchanged. These reductions include
less waiting/queueing, NOT corresponding reductions in wire-transfer service.

## Why overall overhead remains about160ms

Additive owner timeline differences, NEP minus healthy, not mutually exclusive
causal operation costs:

| Timeline region | S2048 ms | S8192 ms |
|---|---:|---:|
| Iteration start through first expert forward envelope | 28.282 | 39.305 |
| Later forward envelopes, total | -6.460 | 14.068 |
| Earlier backward envelopes, total | -1.032 | 0.363 |
| Final backward envelope | 69.394 | 34.681 |
| Last expert backward to optimizer start | 61.734 | 34.731 |
| Optimizer through iteration end | 9.819 | 35.667 |
| Other inter-phase intervals | -3.207 | 0.986 |
| Total | 158.530 | 159.801 |

Final-backward-plus-preoptimizer excess improves131.128 ->69.412ms, but
other regions offset it. This is not evidence that sequence length alone
fixes NEP overall. Healthy timing changes matter: its S2048 second replica
finishes final B59.161ms earlier than full owners, versus only4.003ms atS8192.
Same nodes remove allocation changes but do not guarantee identical runtime
skew. This is one sequential A/B, not a randomized causal estimate.

AtS8192 the30.503ms owner post-optimizer interval partitions temporally into
28.834ms until the last rank's optimizer finishes and1.669ms afterward.
The last optimizer is always a reduced rank in the ten samples (8,9,11,12).
Healthy corresponding intervals are0.115+2.374ms. Thus this is not28ms of
unidentified extra RS transfer. It is mostly alignment with late reduced-rank
optimizer completion. AtS2048 that wait already existed (30.524ms), but
healthy also had a larger post-optimizer interval, masking more of its delta.
The profile does not isolate the last optimizer's internal critical path.

## Reproduction

## Full-owner versus healthy systematic audit

Same S8192 cases and40-49 window, ranks0-5. No new jobs or runtime changes.
The6ms entry and23ms exit skews compare replicas WITHIN NEP. They do not
measure full NEP owner slowdown versus healthy. Non-overlapping temporal
regions sum to159.801ms excess; do not add communication residency to them.

First-forward envelope grows117.715 ->148.961ms. Expert CPU scope sums
34.018 ->33.264ms do not explain this growth. CPU gaps between those calls
grow83.785 ->115.683ms;18.988ms of the NEP span is measured parameter
redistribution host work. Across startup plus first forward, redistribution
CPU totals24.285ms. The code resolves live owner views, packs, broadcasts,
and unpacks in _scatter_nep_distopt_owner_params_to_physical_holders at
nonuniform_ep.py:1699 in the pinned candidate. Async GPU submission does not
make the Python preparation free. Native expert AG operation ranges sum
23.346 ->30.533ms; NEP redistribution ranges38.096ms include waiting and
overlap. These inclusive ranges cannot be added or assigned exactly to the
39.305ms startup-inclusive excess. Attribution to individual exposed host,
copy, and transfer intervals remains partial.

Ordinary forward total excess14.068ms is NOT repeat redistribution: there
are no measured redistribution/gradient-staging calls during those windows.
Second-forward full-owner CPU expert sum34.237 ->32.306ms actually falls,
while CPU time between expert calls83.350 ->92.248ms grows. GPU expert
range sum31.605 ->32.848ms grows only1.243ms versus7.057ms total span.
Most slowdown is outside recorded expert operations. HybridEP/nonexpert
kernel scopes are missing, so this cannot honestly be called confirmed
model-EP contention or expert GEMM inefficiency.

Final-backward submission accounting:

| Measurement | Healthy ms | NEP full ms | Difference ms |
|---|---:|---:|---:|
| GPU expert-to-expert envelope | 111.568 | 146.250 | 34.681 |
| Summed GPU expert operation ranges | 20.183 | 20.539 | 0.356 |
| CPU first expert entry to last expert return | 102.068 | 168.920 | 66.852 |
| Summed CPU expert scopes | 61.298 | 65.330 | 4.032 |
| CPU intervals between expert calls | 40.770 | 103.590 | 62.820 |
| Measured NEP staging inside CPU envelope | 0 | 35.072 | 35.072 |
| Last GPU expert end minus last CPU expert return | 44.909 | 12.668 | -32.241 |

The CPU starts first backward at nearly identical lead35.409/35.339ms.
Additional submission time consumes32.241ms of end-of-backward GPU lead,
with34.681ms appearing as a longer GPU envelope. Staging is a measured major
host cost: _copy_nep_nccl_contexts_to_distopt_grads atnonuniform_ep.py:1336
iterates owner experts/parameter slots and issues individual view/copy calls.
The remaining inter-expert host time includes uninstrumented hooks, packing,
submission, and native execution; exact allocation among them is unknown.
This is not proof that each staging CPU millisecond adds one iteration
millisecond, nor does it exclude native communication contention inside GPU
gaps. It does rule out34.7ms simply accumulating inside the measured expert
operations, whose summed ranges increase only0.356ms.

Post-backward to optimizer-start excess34.731ms, along ordered milestones:

| Component | Healthy ms | NEP ms | Extra ms |
|---|---:|---:|---:|
| Local final expert RS readiness | 0.008 | 3.999 | 3.991 |
| Wait for peer final RS input | 0.271 | 22.216 | 21.945 |
| Both ready to final expert RS completion | 8.107 | 4.276 | -3.831 |
| Final expert RS to final dense RS completion | 1.563 | 8.224 | 6.661 |
| Final dense RS to optimizer start | 5.942 | 11.906 | 5.964 |
| Total | 15.890 | 50.621 | 34.731 |

Post-readiness durations include queueing; healthy/NEP final buckets differ.
Thus -3.831ms is not evidence that NEP has better RS bandwidth. In particular,
the dominant measured positive term is late peer readiness, not wire transfer.

Optimizer GPU envelope grows11.130 ->18.783ms (+7.653), and afterward
2.489 ->30.503ms (+28.014). All-rank matching shows28.834ms of NEP's latter
interval precedes the last reduced rank's optimizer completion, versus0.115ms
healthy. No post-optimizer collective is captured; native iteration logging
uses a barrier. This local interval therefore aligns with global completion,
not28ms of new measured communication. Exact optimizer internal cause is
not resolved: copy_params is nested and cannot be added independently.

These are observed timings plus bounded mechanism evidence, not a complete
kernel-level causal budget. Biggest directly actionable implementation cost
identified here is per-parameter gradient staging during final backward;
the14.1ms ordinary-forward excess and optimizer internal slowdown need finer
instrumentation before proposing a cause-specific change.

## Reproduction Commands

- Launch: scripts/nonuniform/run_nep_sequence_control.sh.
- Entrypoint: scripts/nonuniform/pretrain_nep_sequence_control.py.
- Analyze: `python3 scripts/nonuniform/analyze_nep_sequence_control.py 3073889`.
- Structured result/traces: Lustre slurm_runs/soft_clique_baseline/3073889/,
  sequence_control_analysis/summary.json and seq2048/seq8192 subdirectories.
- No additional training jobs or runtime modifications were made.
