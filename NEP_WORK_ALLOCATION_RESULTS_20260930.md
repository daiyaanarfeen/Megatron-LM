# EP16/12 Work Allocation Results

## Outcome

All five configurations ran successfully without OOM. MBS2 was tested first.
Maximum all-rank peak allocated memory:30.136GiB NEP,11.897GiB healthy.
GB200 capacity184.31GiB; the preflight minimum end-free memory was146.45GiB
for NEP and165.31GiB for healthy. No production NEP implementation changes.

Four latent-MoE layers; H4608/latent1152/F3072/E512/topk6/S1024,TP2/CP1/ETP1.
DistOpt,HybridEP,full MoE graphs,compact placement,disjoint-first,one active lane.
HealthyEP16/16; NEPEP16/12. Actual buckets remain dense1/1,expert2/3(H/NEP)
across all batching configurations. This is not an iso-expert-bucket experiment.

| NEP configuration | Replica batches | True GBS | Healthy ms | NEP ms | Gap ms | Owner parity | NEP ktokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| MBS1 GA4/4 | 32/24 | 56 | 76.075 | 96.633 | 20.558 | 78.723% | 593.418 |
| MBS2 GA2/2 | 32/24 | 56 | 64.242 | 84.450 | 20.208 | 76.071% | 679.029 |
| MBS1 GA4/3 | 32/18 | 50 | 76.075 | 96.717 | 20.642 | 78.657% | 529.381 |

Healthy trueGBS64 in both configurations; throughput861.466 and1020.148ktokens/s.
Parity is the mean of within-allocation healthy/NEP owner-rank0 timing ratios.
Times are means of two complete allocation blocks, six steps per case per block.

- Rebatching MBS1xGA4 to MBS2xGA2 improves NEP throughput14.43%, but healthy
  improves18.42%. The absolute NEP gap remains about20ms; parity falls2.65points.
- Reducing reduced-replica GA4 to3 does not materially improve iteration time.
  Useful aggregate throughput falls10.79%, because fewer samples are processed.
- Therefore this workload does not support underloading the reduced replica as
  a useful optimization. Larger MBS is useful for throughput, not parity here.
- These observations do not establish that every NEP cost is intrinsically fixed,
  or identify the20ms as pure reshard service. They apply to this workload only.

## Profiling And Mechanism Evidence

All ranks have Nsight GPU kernels plus native AG/RS and NEP operation ranges,
process groups,payloads,host submissions and GPU completion. Separate approved
profiler-off companions use the same measured steps15-20 after14warmup steps;
post-profile timings are not substituted. No instrumentation correction is applied.

Profiled total gaps are24.108ms(N1),22.383ms(N2),21.350ms(R), versus the
profiler-off20.558/20.208/20.642ms. Thus the small apparent R improvement in
profiled timing is not reproduced in the timing companions.

For MBS2, profiled owner mean excess is firstF+5.499ms,otherF+0.042,
nonfinalB+0.061,finalB+15.098,outsideF/B+1.658ms. These disjoint phase differences
describe the profiled run, not an exact decomposition of its profiler-off gap.

Paired expert-RS GPU input-ready timestamps (reduced minus full):
N1 -5.451/+7.722ms across the two blocks; N2 -4.473/-3.238ms;
R -19.831/-18.469ms. Paired completion spread is around0.015ms in the first block.
This supports early reduced readiness for N2/R, but not a universal early-arrival
claim for N1. N1 phase allocation changes materially between profiles despite
stable iteration means. Both repeats are shown separately in the phase figure.
Input-ready ranges include dependencies; they are not link service or CPU enqueue.

## Controls And Provenance

- Memory gate3201156 completed0:0 in9m44s; domains07/16,segment4.
- Forward study3201203 was preempted after30m12s. All ten r0 cases form a complete
  block. Three additional completed r1 cases are retained as diagnostics, not
  mixed into the final reverse block; the interrupted case supplies no timing.
- Reverse block3201438 completed0:0 in26m17s on non-preemptibleGB200,segment4.
  Each allocation contains all five cells with healthy controls and both profile
  modes. Compare within allocation before averaging ratios. No hardware mixing.
- Canonical logical weights/layouts match exactly across repeats and allocations.
  Within each topology, native AG/RS payloads and actual buckets are invariant
  across MBS/GA. All25iterations per accepted run have zero skips/NaNs.
- Synthetic routing repeats a randomized exact-balanced512-token route-row unit.
  Expert counts and source/peer assignment counts scale exactly with MBS. This
  controls routing workload statistics, not bitwise training trajectories or the
  association of original dataset sample identities with flattened route rows.
- All12 NEP cross-replica EDP pairs and16 healthy pairs use RDMA; audited local
  owner-transfer groups use P2P. No cross-replica NVLink shortcut is accepted.
- Runtime/harness commit e86d713e7. Trusted placement branch remains untouched.

Artifacts under /lustre/fsw/coreai_comparch_sysarch/darfeen/slurm_runs/nep_cleanup_ab:
3201156,3201203,3201438,and work-allocation-20260930(combined results).
Figures: /lustre/fsw/coreai_comparch_sysarch/darfeen/plots/nep_work_allocation_20260930.
