# Training-workload confounder audit

Date: 2026-09-09 (PDT). Status: historical audit complete within the scope below;
corrected-placement reruns started, not complete.

## Scope and evidence

Audited all 45 successful comparison artifacts (35 allocations, 12 timed MoE
workloads) under `slurm_runs/training_scripts_nep_ab`. Used comparison JSONs,
healthy/NEP driver arguments, original scripts, Slurm accounting/submit lines,
and the cluster's node-to-NVL-block topology. Detailed per-run findings and
runtime arguments are in:

`/lustre/fsw/coreai_comparch_sysarch/darfeen/analysis/nep_workload_audit_20260909/`

Files: `audit.md`, `all_comparisons.json`, `sacct.txt`, `domains.json`,
`topology.txt`. Reproduce using `scripts/nonuniform/audit_training_workload_results.py`.
The two dense scripts (8b and 8b hybrid) have no NEP comparison. Ultra remains a
failed/OOM workload, not a successful parity result; this audit does not turn
missing results into passes. Separate recent a500 diagnostic experiments are
covered by the annex below, not included in the 45-artifact denominator.

## Placement policy

Each replica must reside wholly within one physical NVLink domain, and different
replicas must use distinct domains. `--segment` alone did not establish this in
the historical runs. 36/45 artifacts fail the distinct-domain requirement.
The other nine are consistent with the required layout under the inferred
default Slurm rank mapping, not proven by historical per-rank fabric manifests.
An allocation entirely in one NVL block proves the violation independently of
the inferred mapping.

| Workload | Comparisons | Placement failures | Original rerun allocation |
| --- | ---: | ---: | --- |
| Ladder a315m/1b | 3 | 3 | 4 nodes: 2 + 2 |
| Ladder a500m/2b | 6 | 6 | 4 nodes: 2 + 2 |
| Ladder a770m/4b | 1 | 1 | 4 nodes: 2 + 2 |
| Ladder a1b/7b | 3 | 3 | 4 nodes: 2 + 2 |
| Ladder a2b/14b | 3 | 2 | 8 nodes: 4 + 4 |
| a8b/120b latent MoE | 9 | 3 | 8 nodes: 4 + 4 |
| a3b/30b hybrid | 5 | 5 | 16 nodes: 8 + 8 |
| a3b/30b transformer | 5 | 5 | 16 nodes: 8 + 8 |
| Ladder a3b/30b | 2 | 2 | 16 nodes: 8 + 8 |
| Nemotron Nano | 6 | 6 | 16 nodes: 8 + 8 |
| Nemotron Super | 1 | 0, inferred only | 32 nodes: 16 + 16 |
| a3b GDP latent MoE / Muon | 1 | 0, inferred only | 32 nodes: 16 + 16 |

These are healthy allocation sizes; NEP uses a subset of the second replica's
nodes/ranks. Physical EP degrees are 8/6, 16/12, 32/28, and 64/56 respectively.
Validate EP8 placement first, then EP16, then EP32 before the largest allocations.
Super needs replication because its apparent speedup is noisy. Muon's placement
is plausible, but missing fabric provenance and three timing samples prevent
calling it fully characterized; retain its layer-wise optimizer checkout.

## Confirmed configuration and comparison confounders

- Most original ladder, a3b, and Nano pairs share one NVL domain across replicas.
  Their numbers do not answer the requested cross-domain EDP experiment.
- a8b original job2788305 uses blocks26/30 (different IB leaves), while GA20
  job2877884 uses blocks26/27 (same leaf). Other a8b points share one domain.
  The old sweep therefore mixes network placement with batch-size changes.
- GPU generation changes across tuning points. In particular job2861773 is
  GB200, not GB300 as labeled in the earlier ledger. Canonical a500 is GB300;
  several improved a500 points are GB200. Within-pair hardware is matched, but
  improvement across those pairs is not a clean batch-only comparison.
- Superseded jobs2787055 and2787177 used ETP2 despite source ETP1. They are not
  valid source-configuration results.
- Superseded jobs2786571,2786652,2786661,2786752,2786759 enabled the non-source
  Transformer Engine op fuser. Do not merge them with the later non-fuser runs.
- Source Super/Ultra uses an unavailable cluster-local quantization config;
  the harness substitutes built-in NVFP4. This is disclosed in comparison JSON,
  but is not proof of exact equivalence to the source precision configuration.
- Intentional MBS/GA changes, including below-proportional reduced work, are
  treatment changes, not invisible confounders. Keep local/global batch sizes
  explicit; owner-iteration parity is not equal total tokens per iteration.

No H/N mismatch was found in the recorded values of BF16/FP16/FP8/FP4 recipes,
op-fuser, capacity/padding, CUDA graph settings, optimizer, DDP bucket target,
gradient/parameter overlap, recomputation, router balancing, or energy logging.
This is an argument-level check, not proof that every runtime dependency matched.

## Statistical and attribution limitations

- Most results have only three logged timing samples. Several have CV >3%.
  Nano's121.78% result (2885951) did not reproduce: the extended point2886755
  was92.04%. Super's103.82% includes healthy CV4.51%; neither establishes an
  intrinsic NEP speedup. Report all samples, mean, median, dispersion, and repeats.
- Logged timings and profiled iterations differ (typically8-10 versus5-6).
  Profile-derived milliseconds must not be used to close a different logged gap.
  Profiler overhead and order/warmup effects remain possible.
- Random-logit forced balancing is not an invariant expert/peer split map.
  Per-rank expert launch and participant-readiness variability remains possible
  even when aggregate expert balance is good. Dynamic HybridEP sizing can add
  host synchronization; this is part of the workload unless deliberately changed.
- Collective residency includes participant waiting. It is not payload-only
  transfer time; concurrent waiting is not necessarily useful overlap. Do not
  sum overlapping category deltas as independent causal overheads.
- Equal DDP bucket targets need not produce equal realized expert bucket counts.
  Report actual buckets and bytes. An implementation-induced difference is part
  of NEP cost, not grounds for silently altering the healthy comparator.
- Historical immutable container/package fingerprints, GPU clocks/thermal
  state, background fabric traffic, and synchronized cross-host clocks are
  incomplete. These are unresolved possible confounders, not identified causes.
  No finite retrospective audit can certify that all unknown confounders are absent.

## Recent a500 diagnostic annex

The fixed-node reproduction2979126 and uniform-routing experiment2980905 used
one NVL domain, so both violate the new distinct-domain requirement. Their
mechanistic findings remain observations under that placement, not corrected
cross-domain results. The old below-proportional97.45% mean was94.41% by median;
the fixed-node repeat was94.30% mean/94.25% median. CPU affinity experiments did
not establish a parity fix and are not enabled in the new wrapper. See
`NEP_EXISTING_PROFILE_CLOSEOUT.md` for the underlying observations and limitations.

## Corrected rerun controls

Job2988230 starts with original ladder a500 MBS1/GA3, healthy EP8/8 versus NEP8/6.
Requested nodes: `lyris[0001-0002,0021-0022]`, GB200, NVL blocks01/02, same IB leaf,
`--segment=2`, exclusive nodes. Counted feature requests produced incompatible
Slurm test-only node predictions, so they are not relied upon; explicit node
lists are used. The precise cause of the preview discrepancy is unresolved.

Before training, every rank records GPU UUID/model, fabric UUID/clique/health,
host/domain, CPU affinity, relevant environment, and package versions. A guard
rejects split replicas, shared domains, duplicate GPUs, unhealthy fabric, or
heterogeneous hardware/packages. The pair analyzer also verifies corresponding
replica fabrics and hardware/packages match between H/N. This is startup-only;
no per-iteration hooks or NEP implementation changes were added.

24 iterations, profiler enabled on all ranks for steps10-13, logged timing15-24.
The longer measurement window is deliberate; source model/sequence/MBS/GA is
unchanged for the original point. Profile and logged statistics remain separate.
Each case has a7-minute timeout and the allocation a25-minute cap. If original
parity is insufficient, the same allocation tries MBS3/GA3, then MBS3 with
full GA4/reduced GA3. Both H/N legs use the same hardware for every point.
No source model dimensions, top-k, dispatcher, precision, or NEP code are tuned.

All logs, traces, and result JSONs are under
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/workload_nvl_audit/2988230/`.
Remaining originals will be rerun in increasing allocation size after this
startup path validates. Do not declare the historical parity values reproduced
until corrected-placement results exist.
