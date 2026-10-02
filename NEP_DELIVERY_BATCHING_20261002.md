# Delivery And Rebatching Ablations

Preserve the prior tested branch. This branch is based on0c8ab3e39 plus the
actual-workload harness used in3215521/3215822. Only production file changed:
megatron/core/distributed/nonuniform_ep.py. Native AG/RS code is unchanged.

## Delivery

Actual a8b memory-limited15-MoE hybrid/MTP configuration,H4608/latent1152/F3072,
E512/topk6,MBS2/GA8,EP16/12,TP2/CP1,DistOpt,HybridEP and native graphs.
At S4096 and8192 compare healthy, NEP consumer redistribution, current early
redistribution and early redistribution with two-bucket lookahead.
Local batches128/128 healthy,128/96 NEP; GBS256/224. Identical placement,
useful transfer bytes, optimizer buckets and model dimensions across policies.
Sequence length changes tokens/iteration by design; match healthy per length.

Consumer and early share native AG launch points. They differ only in when
the same dependency/redistribution function is submitted. Lookahead2 launches
the next two expert buckets in canonical order instead of one. The existing
shared delivery stream orders bounded staging-buffer reuse, so additional
transfer buffers are not presumed necessary; measure actual peak memory.
No new host synchronization, polling thread, barrier or native optimizer edit.

Profile-only markers delimit parameter demand and compute-stream readiness,
keyed by expert bucket. Existing all-rank AG/RS/Gather/redistribution, PG,
payload, host-submission and GPU-completion records remain enabled. Bucket
manifests include layer names. Event ranges are not pure transfer service time.

## Batching

At S8192,current early delivery,compare MBS/GA1/16,2/8,4/4 with matched healthy
for each. MBS*GA16 and per-replica local/global batches stay fixed. Routing is
the same repeated exact-balanced unit. This changes tokens per dispatch and
number of dispatches,not total tokens/work allocation. Compare all-forward and
all-backward sums as well as first/other-forward and final/nonfinal-backward:
different GA counts must not be mistaken for phase-efficiency changes.

## Gates

First use real native AG/RS and numerical owner/follower redistribution checks
on four GPUs,all three policies,including force-sync and repeated iterations.
Then EP8/6 four-layer training with every policy and profile validation. Only
after those pass move to EP16/12 actual workload. GB200/GB300 eligible for the
small gate. Use GB300 for actual workload because earlier reserved-memory
measurements fail the conservative GB200 headroom gate.

Memory/failure gate first: MBS4 profiled,then MBS1 companion and profiled.
The previous MBS1 timeout remains unresolved; stop on recurring failure,
preserve evidence,and consult before changing anything outside this ablation.
Do not silently remove MBS1 or replace it with a different model.

Full sweeps use35iterations,matched25-30 after24warmup,all-rank profiles and
separate approved timing companions,two reversed-order repeats. No post-window
substitution. Each replica stays within one NVLink domain; distinct clique IDs
and recorded RDMA cross-replica transport. Canonical initialization,routing,
exact payloads,memory and native optimizer manifests remain acceptance gates.
Poll Slurm every300seconds. All output goes to Lustre and completed plots to
the requested Talos directory. No promise of a favorable ablation result.

## Gate Results And Stop Point

Small job3217536 completed0:0 in14m13s onGB200. All three4GPU numerical modes
and all three EP8/6 training/profile/transfer gates passed. Container isort,
Black and compilation succeeded. The previous tested branch remains untouched.

Actual gate3217613 onGB300 passed profiledM4,P8K,H4 and the M1 timing companion.
M4 peak allocated172.115GiB,reserved216.953GiB,min end-free52.166GiB. M1 timing
companion completed35iterations. Its profiled counterpart again stopped after
iteration25 and timed out in26,matching prior job3215526. Full-replica TP
all-gathers timed out at180s; stacks also show CUDA-event recording during
graph replay. No explicit event-pool exhaustion or OOM. The evidence does not
yet distinguish profiler-caused blocking from an ordering bug exposed by it.
Canceled3217613 after watchdog/stack evidence was saved,elapsed38m43s. No full
sweeps submitted. Requested approval for Nsight-only versus event-only MBS1
profiling isolation with the training implementation unchanged.

An independent benchmark labeling error was found: healthy uses uppercase
native PG descriptors,whereas NEP uses ep_dp/dp_cp. Records were captured,but
the old healthy optimizer fingerprint was empty and H4 deadline markers were
mislabeled dense. Corrected the offline audit and revalidated all44cases in
3215521/3215822 successfully; their timings did not change. Corrected this
branch's marker classifier and future validator after the job stopped. The
saved gateH4 deadline labels remain invalid for expert-specific attribution;
do not rewrite them or use them as valid expert-deadline results. Fresh healthy
profiles are required. These label-only corrections have syntax checks but
have not yet been exercised in a new GPU job.
