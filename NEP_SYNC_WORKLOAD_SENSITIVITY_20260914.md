# First-Forward and Final-Backward Workload Sensitivity

Date: 2026-09-14. Status: initial five-knob ablation complete. No outstanding jobs.

## Results

All use accepted93d313df4, GA3, Flex/HybridEP + DistOpt. Owner means over
ranks0-5, measured40-49. Parent job3057217 completed29m50s, all8 cases passed;
agent jobs3057161/3057162 completed21m50s/28m06s, all14 cases passed. All330
rank profiles parsed successfully. Nothing is reclassified as an irreducible
cost merely because runtime is frozen.

| Matrix / knob | Healthy / NEP owner ms | Measured parity | FirstF / laterF delta ms | FinalB / earlierB delta ms | LastB->optimizer / post-optimizer delta ms |
| --- | --- | --- | --- | --- | --- |
| A anchor start | 709.93 /747.84 | 94.93% | 18.92 /4.47 | 2.82 /0.04 | 2.21 /-0.16 |
| A MBS1 | 452.39 /460.95 | 98.14%* | 7.26 /-4.05 | 2.78 /0.50 | 2.44 /3.04 |
| A anchor end | 714.31 /736.37 | 97.00% | 13.97 /0.37 | 2.61 /0.04 | 2.22 /-0.17 |
| B anchor start | 704.86 /736.07 | 95.76% | 13.59 /4.33 | 2.19 /0.07 | 1.74 /-0.31 |
| B sequence4096 | 434.31 /471.20 | 92.17% | 8.86 /-2.83 | 4.25 /0.07 | 16.40 /9.73 |
| B top-k2 | 621.93 /639.45 | 97.26% | 12.73 /1.17 | 0.70 /0.16 | -0.90 /-0.34 |
| B anchor end | 707.24 /730.65 | 96.80% | 15.59 /3.30 | -1.71 /0.02 | -1.22 /-0.23 |
| C anchor start | 682.48 /730.61 | 93.41% | 24.06 /8.71 | 1.17 /0.11 | -0.22 /0.02 |
| C expert F1536 | 699.09 /772.14 | 90.54% | 35.25 /11.48 | 4.45 /0.18 | 2.23 /-0.01 |
| C shared F3072 | 694.75 /731.81 | 94.94% | 19.40 /4.30 | 2.66 /0.17 | 1.92 /-0.07 |
| C anchor end | 703.13 /727.92 | 96.59% | 16.13 /2.65 | 0.02 /0.13 | -1.78 /-0.03 |

All deltas NEP-full minus matched healthy-full. Later/earlier are per-microbatch
means. These columns are not an additive table: see JSON additive partition for
all intermediate model work, prefix and optimizer. Expert-bounded spans are not
all model kernels or pure compute. Reduced-owner data is retained separately.

*MBS1 is nonstationary: pre/recorded/post parity94.17/98.14/92.12%. Do not
interpret98.14% as a reliable general improvement or subtract29.44ms of
differential recorded/post time as a profiler correction. Agent report records
chronology and ordinary-phase controls. MBS direction remains unresolved here.

## First Forward

Full-owner totals across8 NEP parameter-redistribution calls/update:

| Case | CPU wall ms | GPU operation envelope union ms | CPU calls finished before GPU range start |
| --- | --- | --- | --- |
| B anchor start/end | 8.95 /8.66 | 9.75 /9.61 | 89.2% /93.3% |
| B sequence4096 | 8.39 | 9.76 | 16.3% |
| B top-k2 | 8.48 | 9.52 | 77.5% |
| C anchor start/end | 8.77 /8.92 | 9.74 /10.04 | 92.3% /90.8% |
| C expert F1536 | 8.91 | 17.71 | 93.1% |
| C shared F3072 | 8.65 | 10.15 | 88.5% |

1. Expert width is the clearest payload sensitivity. Routed expert weights
   double, but per-parameter operation count and8 transfer groups remain fixed.
   Parameter-delivery CPU time stays~8.9ms while its GPU envelope grows~8ms.
   NEP's first-minus-later-forward gap grows to23.81ms, versus12.63-15.01ms in
   the two anchors. Healthy's corresponding gaps are near zero. This supports
   increased GPU-side parameter-delivery cost rather than additional host calls.
   It is not a pure network-bandwidth estimate: ranges include copies, transfer
   dependencies and possible peer waiting. Healthy expert rebucketing also
   prevents assigning the whole iteration delta exclusively to payload bytes.
2. Shared width did not demonstrate better hiding of first-forward sync.
   NEP first-minus-later gap15.05ms is within the anchor range12.63-15.01ms,
   and parameter-delivery GPU envelope~10.15ms remains near9.74-10.04ms.
   Measured94.94% parity lies between the two anchors; no robust parity gain.
3. Shorter sequence sharply reduces parameter-launch headroom without changing
   its CPU work or GPU envelope. Nevertheless first-minus-later NEP-minus-H
   gap11.69ms remains within B anchors9.25-12.29ms. The evidence does NOT show
   a large added first-forward delivery penalty here. The reduced parity comes
   with less total work and a larger late-iteration penalty, not just firstF.

## Final Backward and Tail

Sequence reduction leaves gradient bytes and group counts fixed while ordinary
full-owner backward shrinks97.2->51.8ms. Gradient-staging CPU remains~9.6ms;
all staging CPU calls finish ahead of their GPU ranges. CPU-to-GPU start lag
shrinks57->17ms. Thus submission headroom decreases, but the measured staging
calls are not themselves GPU-starved on host submission.

The last-expert-backward-to-optimizer GPU interval is healthy11.09/NEP27.49ms
at sequence4096, versus healthy10.64/NEP12.38ms at anchor start. Last staging
ends2.53ms after last expert backward in NEP, versus1.15ms at anchor start.
Optimizer CPU submission ends6.17ms after last expert backward, still21.32ms
before optimizer GPU starts. This rules out attributing that whole27.49ms
interval to optimizer host submission. It does not identify which uninstrumented
gradient synchronization, remaining model work or stream dependency occupies it.
Additional post-optimizer9.73ms is likewise not proved pure exposed NCCL.

Top-k2 preserves much more other model work: ordinary backward79.1ms, versus
51.8ms for sequence4096, despite fewer token-expert assignments. Staging has
~47.8ms submission lead, and there is no large finalB/optimizer-tail penalty.
This supports whole-microbatch work and native token traffic as relevant
characteristics, not tokens per expert alone. These event ranges do not prove
a specific kernel-efficiency or NCCL-contention mechanism.

Doubling expert width increases staging GPU envelope from6.55-6.74 to9.34ms,
with staging CPU9.26-9.28->9.97ms; all calls remain submitted ahead. NEP final
minus earlier backward grows4.02-4.43->7.58ms, versus healthy2.96-4.54->3.31ms.
This is consistent with greater late synchronization/staging cost when payload
grows. Exact reshard versus EDP attribution requires kernel/collective detail
not present in these low-overhead event profiles.

## Validity and Remaining Limits

- C's anchor parity93.41->96.59% is genuine run-to-run nonstationarity:
  healthy682.48->703.13ms, NEP730.61->727.92ms. Healthy forward accounts for
  much of its shift. Do not compare a knob only with the more favorable anchor.
- Widths deterministically change native bucket thresholds/counts. C baseline
  threshold133822625 elements,3 dense/4 physical expert buckets; expertF1536
  threshold192805025,3 dense/5 physical expert; sharedF3072 threshold141195425,
  4 dense/4 physical expert. Healthy uses those expert groups; NEP replaces
  them with8 logical-owner groups. Requested DDP count remains8 throughout.
  This is a compound execution effect of changing architecture, not an
  iso-bucket payload-only experiment. No claim of equal H/N EDP bucket count.
- A/B bucket signatures are invariant across their knobs within healthy/NEP.
- Recorded/post differential in B is2.13ms anchor start,4.67ms sequence,
  1.15ms topk,-4.71ms anchor end; C ranges1.02-2.89ms. These variations do not
  erase sequence/F-width regressions relative to both anchors, but restrict
  precise causal effect sizes. No profiler-free run or subtraction was used.
- Five knobs tested. H, expert count, depth, routing imbalance and EP ratio
  remain untested in this round. We do not claim a complete sensitivity surface.

Plots copied to https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_sync_workload_20260914/.
Width, sequence/top-k and explicitly nonstationary MBS plots are separate.

## Scope

Treat the accepted implementation as a fixed practical baseline, not a proved
unavoidable overhead floor. No runtime changes. GA remains3 throughout.
Primary question: which workload changes alter first-forward parameter delivery
and final-backward synchronization exposure, distinct from changes to ordinary
expert execution? Later-forward/earlier-backward ablations remain the delegated
agent's primary task; its profiles can answer both questions without duplicate
GPU runs.

Accepted runtime: nep/sync-cpu-20260914 commit93d313df4. NEP SHA256
653a7efd3da53c0ace84c9e32d65eb2faa1f9b17f44b1daf43ace136a3eae0c7.
Same runtime used for healthy and NEP; no rejected gradient-copy optimization.

Anchor: a770 hybrid,15 Mamba blocks,15 MoE blocks,3 attention blocks; H1280,
expert F768, shared F1536,128 true experts/top-k6. Sequence8192, MBS2, GA3;
EP8/8 healthy versus EP8/6 NEP. TP/CP/PP/ETP1. Healthy GBS96, NEP84;
per-replica batches48/48 healthy,48/36 NEP. Flex/HybridEP, DistOpt, BF16,
existing mamba/attn/moe_router graph scopes, requested DDP buckets8.

## Independent Perturbations

- Agent-owned: MBS2->1, sequence8192->4096, top-k6->2, separately. These
  leave parameter/gradient payload unchanged. GA3 and proportional allocation
  preserved. MBS changes sample batch size; sequence changes tokens per sample.
- Parent-owned: expert F768->1536 only. Raises expert arithmetic and routed
  expert gradient/parameter payload. It is not a compute-only experiment.
- Parent-owned: shared F1536->3072 only. Raises shared-expert arithmetic and
  dense/shared parameter traffic, but not routed-expert reshard payload.
  This tests placement of additional non-routed work, not traffic-free compute.

Architecture perturbations can change the native bucket threshold and realized
boundaries even with requested count8. Actual sizes/counts must be reported;
do not normalize these away or attribute all effects exclusively to GEMMs.
H, expert count, depth, routing imbalance and EP-ratio sweeps are not yet run.

## Measurement and Controls

Four-node segment4, GB200 or GB300 first available; fixed nodes/hardware within
each matrix. Both replicas contained in an NVL domain; distinct software cliques
force cross-replica RDMA when physically colocated. Real placement and transport
logs retained. Cached NeMo26.06 container. Per-case six-minute timeout.

60 iterations/case, all-rank targeted CPU/CUDA-event profiles40-49. Compare
owner ranks0-5 and separately reduced owners8-13; healthy uses matching cohorts.
Report10 iteration means, with SD across iterations, not pooled ranks as
independent samples. Pre29-38 and post51-60 detect drift/differential recording
effects; never mix those times into the recorded-window mechanism comparison.

Parent matrix: anchor H/N, expert-F N/H, shared-F H/N, anchor repeat N/H.
An anchor drifting or failing to reach prior parity does not justify silently
changing batch or runtime; assess before interpreting perturbations.

Measured phase endpoints are first/last expert-operation GPU event boundaries,
not the full model forward/backward or pure kernel execution. Include the
iteration prefix, final-backward-to-optimizer gap and optimizer/finalization
separately. The additive temporal partition telescopes exactly to iteration
time, but is not causal attribution to NCCL or compute inside those intervals.
Compare first versus later forward and final versus earlier backward in each
case. Healthy subtraction is a useful descriptive control, not proof that the
entire remaining difference is synchronization.

## Runs

- 3057186: failed before training after1m19s on lyris[0092-0094,0096]. Parent
  pyproject requires Black24, container has26.3.1. Fixed only test preflight
  to use the accepted runtime's Black26 configuration. No benchmark executed.
- 3057217: completed on GB200 lyris[0042-0045], start17:33:13 PDT,29m50s.
  No changes to runtime/workload matrix/profile controls. Monitored every300s.
- 3057161/3057162: agent-owned A/B matrices completed; see agent report for
  exact runtime manifests, nodes, transport validation and per-iteration rows.

Artifacts: /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/
soft_clique_baseline/JOB/TAG/cupti/ep8/scaling_ladder__a770m_4b_moe_215b/
{healthy,nep}/JOB/. Per-rank launch_manifest records exact post-override args,
runtime hash and environment. Summary output: JOB/sync_workload_analysis/.

New harness: scripts/nonuniform/run_nep_sync_workload_20260914.sh and
pretrain_nep_sync_workload.py. New analyzer: analyze_nep_sync_workload.py.
Argument tests confirm each perturbation changes one token only. Analyzer
validated against historical3054530 anchor (95.7924% owner parity); that is a
self-check on older data, not evidence for the current implementation.
