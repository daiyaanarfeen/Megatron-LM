# WITHDRAWN: NEP Placement and Transfer Scheduling Ablations

## Status: Invalid for Paper Evidence

Withdrawn September 29, 2026 at the user's request. All published performance
figures, tables, and conclusions from this set are removed. The recommendation
of compact over spread placement is retracted, not merely qualified.

The implementation broadcast an owner's entire offloaded-expert buffer to every
participating follower. Each follower received experts it did not execute. This
violates the required transfer policy: only the experts needed by a GPU pair
should be transferred between that pair. Both compact and spread use this path;
removing only spread results would not address the invalid implementation.

Scheduling measurements made with this broadcast path are also withdrawn as
evidence for the intended implementation. They must be repeated after correction.
Passing value tests and finite training did not validate communication minimality.

## Affected Experiments

| Job | Experiment | Status |
|---|---|---|
| 3181167 | EP8/6 placement and parameter order | Withdrawn |
| 3181303 | EP8/6 release timing and gradient order | Withdrawn |
| 3181315 | EP8/6 compute-rich coverage | Withdrawn |
| 3181288 | EP16/12 placement and parameter order | Withdrawn |
| 3181404 | EP8/6 real-routing coverage | Withdrawn |
| 3181439 | EP16/12 release timing and gradient order | Withdrawn |

The paired healthy controls remain in the raw records but do not salvage these
comparisons. Correctness gates3181152 and3181250 are historical execution/value
tests only, not proof of a correct destination-specific transfer protocol.

## Provenance Only

- Raw logs, traces and numerical summaries remain under
  `/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/` for debugging.
- Each affected run root is marked with `WITHDRAWN.md` and `result_status.json`.
- Historical September28 journal entries are superseded by the September29
  withdrawal and must not be cited as active results.
- Experimental commits `1f084d5dc` and `3225f09a1` remain reproducibility records,
  not endorsed implementations. The trusted rollback branch was not modified.
- The report generator now publishes withdrawal notices instead of regenerating
  invalid figures. Existing SVG/JSON download URLs are replaced with notices.

## Before New Measurements

Implement destination-specific transfers, validate per-pair expert identities and
required payload bytes, then rerun placement and scheduling comparisons. Parallel
transfer streams remain an open ablation; these results do not justify ruling
them out. No runtime implementation changes or new jobs were made for this
withdrawal request.

Published withdrawal:
https://sc.talos.nvidia.com/view/home/scratch.darfeen_coreai/nep_placement_ablations_20260928/index.html
