# Gather-Only CTA Cap

Parent `nep-norm-headroom-20261006` at be5dcb32f. No training runtime edits.

User approved trying the proposed16-CTA cap on `nep_owner_gather` only.
Native `--nccl-communicator-config-path` loads the two-line YAML; no global
NCCL resource variables, native RS/AG settings, or scheduling changes.

## Controls

- Healthy EP16/16; reference/capped NEP EP16/12; TP2, CP1, ETP1.
- Saved memory-limited a8b15-MoE hybrid/MTP, H4608/latent1152/F3072,
  E512/topk6, MBS2/GA8/S8192. Healthy localBS128/128, NEP128/96.
- DistOpt, Flex/HybridEP, native MoE graphs, exact randomized balanced routing.
- Five expert buckets, three dense buckets, norm margins forward16/backward32,
  compact placement, ascending transfer order, one stream, prefetch1.
- Eight GB300 nodes, segment4, each replica in one NVL domain, cross-replica
  optimizer transport audited as RDMA. GB200 failed prior memory-headroom gate.
- Repeat0 GBASE/HGC/G16, profile then timing companion; repeat1 reverse order,
  companion then profile. Twelve cases,35iterations, matched measured25-30.
- Preserve all-rank AG/RS/Gather/parameter redistribution profiles, payloads,
  host submission and GPU completion. Do not substitute post-profile timing.

## Acceptance

Verify actual Gather kernel grids shrink from32 to at most16 while norm grids
stay360, and unrelated process-group settings/payloads remain unchanged.
Compare owner iteration times, phase overheads, norm admission gaps, propagated
model-EP/TP waits and Gather residency. Residency includes peer waits and
queueing, not just transfer service. Reject a local improvement that worsens
absolute iteration time. Preserve all repeat results, including anomalies.

Gather16 may still exceed the32-SM headroom when both expert and dense RS run.
Do not change native RS caps without consulting the user. No default cap is
promoted unless evidence supports it; original branches remain rollback points.

## Results

Job3254182 completed12/12 cases, exit0, elapsed1:06:40. Experiment checkpoint
9f296a7fb. Actual placement: theia0170/0171/0175/0180 in nvlblk26 and
theia0280/0282/0283/0284 in nvlblk32. Both repeats pass cross-replica RDMA,
payload/routing/initial-weight, memory, and all-rank trace gates. The existing
parameter-freshness/follower/reuse/force-sync preflight also passed. This is
not a claim of bitwise equality of final dummy-training weights.

Owner-rank0-11 means, matched iterations25-30 after24 warmup iterations,
two reversed-order repeats. Parity = healthy time / NEP time.

| Mode | Healthy ms | Default Gather ms | Gather16 ms | Default parity | Gather16 parity |
|---|---:|---:|---:|---:|---:|
| All-rank profile | 3198.064 | 3261.268 | 3252.751 | 98.062% | 98.319% |
| Authorized timing companion | 3154.082 | 3213.746 | 3201.785 | 98.143% | 98.510% |

Absolute NEP improvement:8.517ms (0.261%) profiled,11.961ms (0.372%) companion.
Both profiled repeats improve:9.037/7.998ms. Both companion repeats improve:
6.365/17.558ms. Followers and reduced-replica mean iteration times improve by
the same amounts (all cohort means within0.05ms). No post-profile substitution.
Profile-minus-companion differential versus healthy is3.540ms default and
6.984ms capped; the profile therefore understates the observed companion gain.
These paired differences include run variability, not a precise subtraction
of instrumentation cost from individual kernel measurements.

| Profile repeat | Healthy mean (SD) ms | Default NEP mean (SD) ms | Gather16 NEP mean (SD) ms |
|---|---:|---:|---:|
| 0 | 3200.232 (4.744) | 3257.585 (1.324) | 3248.548 (4.685) |
| 1 | 3195.897 (6.057) | 3264.951 (4.187) | 3256.954 (3.852) |

| Companion repeat | Healthy mean (SD) ms | Default NEP mean (SD) ms | Gather16 NEP mean (SD) ms |
|---|---:|---:|---:|
| 0 | 3154.611 (3.750) | 3214.183 (2.122) | 3207.818 (4.208) |
| 1 | 3153.554 (4.392) | 3213.310 (1.501) | 3195.752 (6.818) |

## Mechanism And Tradeoff

Measured-iteration kernel grids were audited on all28 NEP ranks in both
repeats. Owner Gather32->16CTAs; follower kernels follow min(original,16),
including their smaller10/18CTA payloads. Norm stays360CTAs. Unrelated
collective kernel-grid/count distributions are unchanged rank by rank:
parameter redistribution, native expert/dense RS/AG, and other native groups.
There is no runtime diff against the parent branch.

| Owner phase, profile ms | Healthy | Default NEP | Gather16 NEP | Capped minus default |
|---|---:|---:|---:|---:|
| First forward | 175.999 | 194.806 | 194.595 | -0.211 |
| Other forwards | 1129.523 | 1130.196 | 1128.151 | -2.045 |
| Nonfinal backwards | 1602.802 | 1605.725 | 1601.984 | -3.741 |
| Final backward | 238.930 | 270.141 | 262.114 | -8.027 |
| Outside forward/backward | 51.169 | 60.731 | 66.263 | +5.533 |

Phase windows and native iteration timers differ slightly in boundary
instrumentation; they are not pure compute durations. Small changes in
unmodified ordinary phases should not all be attributed to the cap.

- Owner norm no-native-work gaps per final backward:3.801->0.883ms (-76.8%).
  Norm kernel residency stays8.096->8.141ms, unlike the previous margin64
  experiment that slowed normalization on every microbatch.
- Full-replica ranks0-15 mean norm-gap sum:3.252->1.160ms. Graph gaps>1ms:
  180->46; maximum7.058->2.940ms. Ordinary-backward norm gaps remain0.254ms.
- This does not eliminate stalls everywhere: follower norm-gap sum increases
  1.606->1.991ms. Healthy final-backward gaps also vary across repeats; its
  combined full-replica mean is0.790ms, not universally zero.
- Owner summed Gather kernel residency:30.770->41.973ms/iteration.
  Expert RS116.834->120.287ms, dense RS27.344->31.495ms; parameter
  redistribution14.462->14.486ms. These include peer waiting and overlapping
  kernels, so they must NOT be added to phase/e2e overhead or called transfer
  service times. They show the cap has a real communication tradeoff.
- Final backward improves8.027ms while outside-F/B grows5.533ms. The cap
  helps but does not remove the overall NEP gap. Exact attribution of that
  outside-F/B increase is not established by these aggregate residency sums.

## Decision And Artifacts

The approved Gather-only cap is a modest net win on this workload, with the
same direction in all four comparisons. Keep this as a workload-specific
configuration on the isolated branch; do not claim a universal optimum or
silently cap native RS/AG. No additional runtime change or job was submitted.

Root: `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab/3254182`.

- `gather-cta-summary.json`: timings, phases, norm gaps, collective residency.
- `actual-summary.json`, `work-controls-audit.json`: all cases and transport.
- `gather_cta/`: per-rank cached analysis, reduced-grid audit, analysis log.
- Per-case `cupti_profile/`, `nsys_profile/`, `actual-validation.json`.
- `analysis_source/`: offline analyzer and its imported dependencies.

Analyzer: `scripts/nonuniform/analyze_nep_gather_cta.py` in the main workspace,
archived with the results. It matches collective NVTX ranges to runtime launch
correlations and kernels and selects only iterations25-30. Raw captures also
contain iteration24; those kernels are excluded from the reported residency.
