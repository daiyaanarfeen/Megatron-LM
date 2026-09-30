# Randomized Routing Controls

Benchmark-only continuation of the controlled NEP placement study. Native NEP,
dispatcher and optimizer implementation files are unchanged by this patch.

- `nep_balanced_routing.py` builds seeded degree-preserving edge-swap maps.
  Every token selects distinct experts and every expert receives exactly the
  same number of assignments. It does not claim uniform sampling of all regular
  graphs. Unique-token/destination counts are recorded, not forced identical.
- `nep_randomized_controls.py` caches routing before graph capture/timing and
  initializes expert weights by logical expert identity using native initializers.
  Startup synchronization must preserve exact values. Parameters and actual
  dense/expert bucket layouts are recorded on every rank.
- `validate_nep_randomized_controls.py` rejects mismatched paired controls.
- `nep_all_ready_replay.py` calls the actual parameter transfer path with all
  owner inputs ready, before optimizer construction. It excludes AG/RS, model
  compute and token dispatch, checks exact final parameters, and profiles every
  rank. Its latencies include packing, launch, waits and unpacking, not just wire
  service. Replay results are not end-to-end training parity.

## Validation, September 30, 2026

| Job | Check | Result |
|---|---|---|
| 3199910 | CPU, CUDA counts/STE and CUDA graph replay | Pass |
| 3199948 | EP8/6 compact/spread, 25 training iterations each | Pass |
| 3200181 | Six-policy EP8/6 all-ready transfer replay | Pass |
| 3200350 | EP16/12 compact/spread, 25 training iterations each | Pass |

Both training gates preserve all4096 logical expert tensors, dense weights,
routing maps, bucket layouts and native collective manifests. Every parameter
source/destination split matches. All-ready policies also preserve exact values
and aggregate bytes. Two-lane transfer kernel concurrency was confirmed on GPU.
Final performance ranking requires counterbalanced repeats and matched-window
profiling-overhead checks; validation timings alone are not final study results.

Artifacts are under `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/`:
`nep_randomized_controls/3199910` and `nep_cleanup_ab/<job>` for the other jobs.
Each training root contains `randomized-controls-validation.json` and
`randomized-gate-analysis.json`; replay contains `all-ready-analysis.json` and
`all-ready-kernel-concurrency.json`. Shared launcher snapshots are archived there.

The EP16 gate used the EP8 wrapper with `sbatch --nodes=8 --segment=4
--export=ALL,GRAPH_FULL_EP=16,GRAPH_REDUCED_EP=12,NCCL_PXN_DISABLE=0`.
Both replicas must remain within individual NVLink domains. Resource selection
is separate from the routing seed; compare policies within the same EP size.

## Counterbalanced Study

`run_nep_policy_study.sh` and `validate_nep_study.py` add per-case fail-closed
checks and unique policy/seed/repeat outputs. Production transfer code is unchanged.
Replay diagnostic companions can disable instrumentation with the explicitly
authorized `NEP_CONTROLLED_PROFILE=off`; CUDA elapsed timers and exact final
parameter checks remain. Profiling stays enabled by default.

Jobs3200495/3200606/3200857 completed all cases with controls passing. The final
profiled reference replay nevertheless has an unexpected host-submission stall
on ranks9/10. Further experiments/causal attribution were stopped per user policy;
no samples were silently dropped. See the September30 journal for measurements.
Stable profiler-off companion results do not establish the underlying cause of
that profiled reference anomaly.
