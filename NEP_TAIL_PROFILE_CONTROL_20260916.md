# Final-sync diagnosis with broad profiling removed

## Result

**The late reduced-peer bottleneck persists without the broad profiler.**
This does not claim that prior profiler configurations never penalized NEP.
It validates the current workload and current prefetch implementation.

Job **3073618**, COMPLETED 0:0, September16,10:14:47-10:38:26PDT,23:39.
GB200 `lyris[0037,0042,0044-0045]`, segment4. Both replicas physically in
nvlblk03 with software NVLink cliques1000/1001 enforcing cross-replica RDMA.
Sequential sparse healthy/NEP, none healthy/NEP, full healthy/NEP on one allocation.
Every case completed60iterations without skips/NaNs; timings use40-49 only.

Same E512/S2048 a770-family workload: H1280,expert F768,shared F1536,top-k6;
15MoE/15Mamba/3GQA; MBS2,GA3; healthyEP8/8 versusNEPEP8/6; TP/CP/PP/ETP1;
replica batch48/48 versus48/36; DistOpt/Flex/HybridEP/local graphs unchanged.
Reference93d313df4d55 and candidate51fdbbb11 unchanged. No runtime code edits.

## Timing control

Means +/- sample SD over the same10 logged iterations in each case:

| Instrumentation | Healthy, ms | NEP, ms | Parity |
|---|---:|---:|---:|
| None: profiler and its wrappers never installed | 661.22 +/-6.77 | 908.94 +/-8.27 | 72.75% |
| Sparse final-sync markers only | 645.80 +/-3.90 | 897.54 +/-11.48 | 71.95% |
| Previous full CUDA-event operation profiling | 654.11 +/-3.92 | 893.42 +/-9.48 | 73.21% |

None/sparse use `--profile` removed; all-rank logs confirm `profile=False`.
The diagnostic entrypoint asserts that neither broad event-capture module nor
collective-capture module was imported. None installs no training hooks and
produces no traces. Sparse installs only7 markers/healthy owner,9/NEP owner,
5/NEP follower per measured iteration. No in-window event queries or host
synchronization; final RS completion uses observer streams with no dependency
back into training streams. No extra collectives are inserted.

Sparse timing is1.25% lower for NEP and2.33% lower for healthy than the fresh
uninstrumented cases. Thus these runs do not isolate a precise marker cost from
ordinary between-run variation, but there is no observed large NEP marker
penalty. Sparse NEP pre/measured/post means898.24/897.54/898.53ms also show no
slowdown when its markers activate. Pre/post values are diagnostics only, not
substitutes for measured timings or a profiler correction.

Do not interpret the full-profile run being slightly faster than the unprofiled
run as proof that profiling accelerates training. These are sequential runs,
not an order-randomized estimate of small instrumentation costs. The relevant
result is that removing profiling does not remove the large NEP slowdown.

## Final-sync milestones

All six full-owner/reduced-peer pairs (0-5 paired to8-13), iterations40-49.
Times relative to each full owner's last expert backward completion:

| Milestone, ms | Sparse healthy | Sparse NEP | Full-profile NEP |
|---|---:|---:|---:|
| Local final expert RS input ready | 0.697 | 54.091 | 71.583 |
| Peer last expert backward complete | 6.578 | 83.127 | 100.182 |
| Peer final expert RS input ready | 7.359 | 86.653 | 103.472 |
| Final expert RS complete locally | 15.183 | 90.855 | 107.854 |
| Final dense RS complete locally | 19.296 | 99.835 | 117.499 |
| Optimizer starts | 25.310 | 111.890 | 129.573 |

Sparse NEP local final Gather/staging completes at +54.090ms, effectively the
same as its final RS input readiness. From there:

- **32.562ms waits for reduced-peer input readiness.** Across60 pairs/iterations
  this is always positive, range21.993-40.211ms.
- **4.202ms remains after both final RS inputs are ready**, range3.783-5.459ms.
  This is post-readiness completion time, not pure transport/kernel service.
- Waiting is88.6% of that final RS interval. Full profiling gives31.889ms wait
  and4.382ms post-readiness completion, the same qualitative and quantitative
  result for this dependency.
- The reduced replica already starts final backward56.379ms later (healthy
  peer5.294ms); it finishes83.127ms later (healthy6.578ms). The extra final
  peer lateness is76.549ms against86.580ms additional owner post-backward tail.
- Final expert backward spans92.303ms healthy versus163.920ms NEP full owner;
  reduced-peer span190.667ms. These are elapsed spans, not GEMM busy time.

Sparse cross-rank final-RS completion skew is -0.246..+0.053ms for NEP,
far smaller than22-40ms readiness waits. Clock calibration occurs before
measurement; a small clock drift is possible. No claim that each microsecond
is a separately measured transport cost.

This confirms option1 as the dominant directly resolved final-sync limiter.
It does not fully explain why the reduced replica is late or partition all
of final backward into compute, host spacing, native EP stalls, and Gather.
Do not relabel queued Gather ranges as pure transfer cost.

## Reproduction and failed attempt

- Launcher: `scripts/nonuniform/run_nep_tail_control.sh`.
- Diagnostic: `scripts/nonuniform/pretrain_nep_tail_control.py`.
- Analyze: `python3 scripts/nonuniform/analyze_nep_tail_control.py 3073618`.
- Initial attempt3073579 finished sparse healthy but its post-run checker
  expected full profiler files. Fixed only the diagnostic mode label to use
  the existing no-CUPTI control allowance; no fake profiler files were created.
- Runtime worktrees remain unchanged and clean. Syntax checks and all-rank
  diagnostic call-count assertions pass.

Artifacts under:
`/lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/soft_clique_baseline/3073618/`.
Each `<mode>_<case>/cupti/ep8/scaling_ladder__a770m_4b_moe_215b/<case>/3073618/`
contains its log/manifests and either `tail_markers/`, `cupti_profile/`, or no
instrumentation output. Summary: `tail_control_analysis/summary.json`.
