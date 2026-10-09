# Compact a8b parameter delivery

## Implementation

- Branch: `nep-param-compact-20260917`, commit `d638286d7`.
- Worktree: `.worktrees/nep-param-compact-20260917`.
- Rollback/reference: unchanged `.worktrees/nep-param-prefetch-20260915`,
  commit `51fdbbb11`.
- Only the DistOpt parameter-transfer plan and its unpack operation changed.
  The owner keeps its own parameters local; the broadcast packs only true
  experts assigned to other physical holders. Compact offsets are cached.
- Collective groups/order, transfer stream, AG prefetch, buffer reuse,
  gradient Gather/RS, optimizer and model workload are unchanged.
- Estimated full-follower receive payload for a8b: 68.479 -> 16.987 GB per
  iteration. This is a static-layout estimate, not measured performance.

## Validation

Job `3088005`, GB200, four GPUs, completed 0:0 in 2m39s:

- Real NCCL compact delivery: E16 EP4/2, E23 EP4/3, E13 EP4/2, E4 EP4/3.
- Exact parameter values verified for every physical expert and parameter
  slot, repeated across three iterations and three shared-buffer groups.
- Includes nondivisible placement, singleton groups and rebound proxy storage.
- Existing native AG/RS + asynchronous prefetch test passed, including
  force-sync, independent GPU work, native model-EP work and repeated finish.
- Four focused pytest tests passed. Imports formatted with `uv run isort`,
  affected Python files formatted with Black. Runtime compile checks passed.
- All-rank operation profiles: Lustre `analysis/compact_smoke_3088005`.

## Original-depth GB300 comparison pending

Job `3088020` requests eight GB300 nodes, segment4, 30 minutes. NEP then
healthy, 45 iterations each; MBS2, GA8, sequence8192, EP16/12 versus16/16,
DistOpt/Flex, graph and bucket settings unchanged. Profile iterations25-34;
post-profile36-45. Runtime SHA256:
`677332c72a1242ccd61334901c60b37c34500f021ecc96a527ffeec09b28966d`.

At approximately 10:00 PDT September17, job pending with no ETA. Slurm's
active `maint` reservation covers all 288 GB300 nodes, restricted to admin
accounts; `gb300_checkout` also covers 245 nodes. These have placeholder
year-long end dates, not a reliable maintenance-completion estimate.
No valid after measurement is available yet. Monitoring agent Euclid
(`01a0b04f-9355-72b2-8658-1d54398d75a4`) owns five-minute start polling.

The comparison to fill from matched profile windows is:

| Interval | Healthy before ms | NEP before ms | Healthy after ms | NEP after ms |
| --- | ---: | ---: | ---: | ---: |
| Startup + first forward | 287.7 | 673.8 | Pending | Pending |
| Other seven forwards | 1716.7 | 1800.5 | Pending | Pending |
| First seven backwards | 4743.8 | 4858.4 | Pending | Pending |
| Final backward | 698.3 | 810.3 | Pending | Pending |
| Optimizer/tail/transitions | 997.9 | 1038.6 | Pending | Pending |
| Iteration | 8444.4 | 9181.5 | Pending | Pending |

## Resume analysis

After job completion, verify both cases reach iteration45 and collect all
32 healthy / 28 NEP rank files under Lustre
`slurm_runs/soft_clique_baseline/3088020/mbs2/cupti/ep16/a8b_120b_latentmoe_1t`.
Check domain placement and cross-replica transport before comparing timings.

```bash
python3 scripts/nonuniform/analyze_a8b_current_overheads.py --job 3088020 --tag mbs2
```

For delivery-chain matching, import `analyze_a8b_delivery_chain`, set its
`ROOT` to job3088020's `mbs2_overhead_analysis`, then call `main()`.
Compare against job3078207 using the same full-owner ranks0-11 and ten
iterations25-34. Report post-profile timings36-45 separately. Do not claim
an end-to-end improvement based only on the expected payload reduction.

## GB200 matched before/after result (September 17)

Job `3089470` completed both versions on the same seven GB200 nodes, with
45 iterations each and all 28 rank profiles. The eight-node allocation used
segment4. Both replicas were inside nvlblk01 but had distinct MNNVL clique
IDs; NCCL logs confirm cross-replica traffic used NET/NCCL RDMA Plugin/GDRDMA,
not cross-replica NVLink. Only depth was reduced to fit GB200: 15 MoE layers
including two MTP layers. A 20-MoE attempt OOMed. H4608/latent1152/F3072,
512 experts/topk6, EP16/12, TP2/CP1, MBS2/GA8/seq8192, GBS224, DistOpt/Flex,
graphs and requested DDP buckets8 are unchanged between versions.

| Full-owner interval | Before ms | Compact ms | Change ms |
| --- | ---: | ---: | ---: |
| Startup + first forward | 337.87 | 226.83 | -111.03 |
| Other seven forwards | 883.78 | 887.57 | +3.79 |
| First seven backwards | 2302.27 | 2354.90 | +52.64 |
| Final backward | 475.82 | 430.17 | -45.65 |
| All backwards | 2778.09 | 2785.08 | +6.99 |
| Profiled iteration, log mean25-34 | 5024.51 | 4913.90 | -110.61 |
| Post-profile iteration, log mean36-45 | 5004.24 | 4873.58 | -130.66 |

The all-backwards row is a subtotal, not an additional phase. Phase ranges
average owners0-11; log means are rank0's native iteration timer. The latter
excludes a small amount of logging/profiler-boundary host time. Standard
deviations of logged profile iterations are7.53/10.86ms; post-profile9.39/7.29ms.
Improvement is2.20% profiled and2.61% post-profile. No fresh healthy case was
run in this allocation, so this is not a new matched healthy-parity claim.

### Why the previous final-backward increase happened

The earlier comparison used before3088971 and after3089211. Full-owner final
backward grew416.96->592.91ms, but full-owner ordinary backward got faster
335.90->322.93ms per microbatch, while reduced ordinary backward got slower
345.55->352.53ms. Reduced-minus-full final-backward arrival skew therefore
grew11.82->204.93ms. Full ranks waited for reduced gradients; the final phase
records accumulated earlier-microbatch skew, not just that phase's work.

This run-to-run native execution-rate change is visible in measured expert
CPU thread time, not merely inferred from a GPU gap. The critical full-rank
expert backward+recompute CPU time fell89.24->79.99ms per ordinary microbatch;
the critical reduced rank increased99.08->104.36ms. Critical rank identity
changed as well. CPU core placement/frequency was not recorded, so the exact
hardware/OS reason for this CPU-time variation is not established. Do not
claim a particular clock, NUMA, or scheduler cause from these profiles.

The same-node repeat does NOT reproduce a compact-transfer backward
regression: final backward decreases45.65ms. Full ordinary backward instead
increases328.47->336.11ms, reduced stays344.99->345.74ms, and final arrival
skew decreases83.24->28.51ms. Thus ordinary native execution pace varies even
between process launches on the same nodes. Total full backward differs by
only6.99ms (0.25%), and the first-forward gain survives end-to-end.

Matched owner RS sums after the latest participant's input-ready event were
112.40->111.91ms in the original comparison and109.76->109.88ms in the repeat.
These are summed operation remainders, NOT wire times or additive iteration
overheads. Corresponding summed participant waits were45.41->489.65ms and
210.09->79.44ms, respectively. RS service growth does not explain the jump.

### Checks and remaining scope

- All28ranks/10steps have identical operation counts and identical non-param
  collective payload multisets. Source-counted parameter broadcast payload
  decreases109.566->27.179GB/iteration, without changing AG/RS/Gather.
- Both cases complete45/45iterations, no skipped or NaN iterations. Existing
  exact-value compact-transfer and native DistOpt integration smokes passed.
  Benchmark loss trajectories are NOT bitwise equivalent: repeated runs of
  the unchanged implementation also diverge after early roundoff differences
  on this unstable dummy-data workload. Do not claim loss equality as a test.
- Retain compact transfer at `d638286d7`; no new backward implementation fix
  was made. Reduced EP still has43 expert slots/GPU versus32 at full EP, and
  its native expert execution/host work can remain the slower path. Optimizing
  that is separate from parameter broadcast packing; this result does not
  prove that remaining overhead is intrinsically unfixable.
- Raw profiles and summaries are under Lustre
  `slurm_runs/soft_clique_baseline/3089470/gb200_{before,after}_overhead_analysis`.
  `backward_skew_from_3089470.json` records per-step/rank readiness and CPU
  evidence. Reproduce with `scripts/nonuniform/analyze_a8b_backward_skew.py
  --before-job 3089470 --after-job 3089470`.
