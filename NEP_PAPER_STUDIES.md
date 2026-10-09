# NEP Paper Studies

## Start Here

- [Paper introduction](NEP_PAPER_INTRODUCTION.md). Bracketed aggregate numbers are placeholders.
- [Contributions and algorithm](NEP_CONTRIBUTIONS.md).
- [Experiment journal](NEP_APPROACH_A_JOURNAL.md). Historical entries include superseded and failed runs; do not treat every entry as accepted evidence.
- Analysis, profiling adapters, plotting, and launch code: [scripts/nonuniform](scripts/nonuniform).
- Earlier frozen workload inputs: [experiments](experiments).

This branch archives paper materials without merging experimental implementation changes
into the implementation branch. The separate snapshots below preserve the corresponding
runtime code, test harnesses, and reports. Snapshot commits are exact; launchers retain
their original cluster/worktree paths and may need path configuration on another machine.
This is a research artifact archive, not a portable one-command benchmark package.

## Main Evidence

| Study | Report |
|---|---|
| Placement and work allocation | [NEP_REAL_ABLATION_RESULTS_20261002.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/0cee1d14ad9f3113f43dad6cb47ae80530ffd566/NEP_REAL_ABLATION_RESULTS_20261002.md) |
| Fixed-work MBS/GA | [NEP_DELIVERY_BATCHING_20261002.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/2280f19114c1c03408a4bb1cd0f67b2b0f95c735/NEP_DELIVERY_BATCHING_20261002.md) |
| AG prefetch and redistribution release | [NEP_DELIVERY_FACTORIAL_20261005.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/11e2388c6d5e17556dd9c463c177f306121cdc3f/NEP_DELIVERY_FACTORIAL_20261005.md) |
| Sequence length at GA8 | [NEP_HIGH_PARITY_ABLATION_20261005.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/7c23ac8e257fc1ad574244e4accd623c64de4cb8/NEP_HIGH_PARITY_ABLATION_20261005.md) |
| Expert bucket sensitivity | [NEP_EXPERT_BUCKETS_20261006.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/a6f90f29e263777e28757309ee7672c828781fda/NEP_EXPERT_BUCKETS_20261006.md) |
| Gather interference | [NEP_GATHER_CTA_20261006.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/83c1fd9c09c51e5cb553ead76e3c3e39186cc595/NEP_GATHER_CTA_20261006.md) |
| Expert count at GA8 | [NEP_EXPERT_COUNT_20261007.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/fc815d6069d5df185b37c99ce7d780e2586c40d5/NEP_EXPERT_COUNT_20261007.md) |
| Expert count at GA2/GA4 | [NEP_EXPERT_COUNT_GA_20261007.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/56e567f1cad19a301fe6b283315bb3d1b14a31d8/NEP_EXPERT_COUNT_GA_20261007.md) |
| Sequence length at GA2 | [NEP_SEQ8K_GA2_20261008.md](https://github.com/daiyaanarfeen/Megatron-LM/blob/70877f7d8b4f8f4519c7feaace39414ffd496a66/NEP_SEQ8K_GA2_20261008.md) |

## Interpretation

Use owner-rank iteration parity, not aggregate-throughput equivalence. Keep profiled
and authorized matched-window timing companions separate. The sequence and expert-count
studies use different configurations; they are not one pooled scaling curve.
Topology, payload, routing, warmup, and validation details are in each report.
Historical studies of broadcast-based spread placement were superseded by destination-
specific transfer studies; their presence in this archive does not validate those results.

## Data Locations

Large artifacts remain outside Git under:

`/lustre/fsw/coreai_comparch_sysarch/darfeen`

Reports record job IDs, trace paths, summary JSON paths, and existing Talos plot links.
This push does not upload raw traces, checkpoints, logs, or rendered plots. Reanalysis
that reads those files still requires access to Lustre or a separately transferred copy.

## Exact Study Snapshots

The corresponding fork branches use the `paper-studies/20261009/` prefix.
All prior implementation history is retained; no force-push or upstream push is needed.

| Snapshot branch | Commit |
|---|---|
| [paper-studies/20261009/nep-ag-red-overlap-20260922](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-ag-red-overlap-20260922) | `ccb4b9726201a02239c1bc4557ef3094b7484439` |
| [paper-studies/20261009/nep-buffer-cleanup-20260921](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-buffer-cleanup-20260921) | `ee28f09fff8c7af63870c44ad7f205c80e744349` |
| [paper-studies/20261009/nep-delivery-batching-20261002](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-delivery-batching-20261002) | `2280f19114c1c03408a4bb1cd0f67b2b0f95c735` |
| [paper-studies/20261009/nep-delivery-factorial-20261005](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-delivery-factorial-20261005) | `11e2388c6d5e17556dd9c463c177f306121cdc3f` |
| [paper-studies/20261009/nep-expert-buckets-20261006](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-expert-buckets-20261006) | `a6f90f29e263777e28757309ee7672c828781fda` |
| [paper-studies/20261009/nep-expert-count-20261007](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-expert-count-20261007) | `fc815d6069d5df185b37c99ce7d780e2586c40d5` |
| [paper-studies/20261009/nep-expert-count-ga-20261007](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-expert-count-ga-20261007) | `56e567f1cad19a301fe6b283315bb3d1b14a31d8` |
| [paper-studies/20261009/nep-gather-cta-20261006](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-gather-cta-20261006) | `83c1fd9c09c51e5cb553ead76e3c3e39186cc595` |
| [paper-studies/20261009/nep-graph-launch-order-20260921](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-graph-launch-order-20260921) | `8abb29fca8596737b4ec8acbd7a6d4719725a080` |
| [paper-studies/20261009/nep-high-parity-delivery-20261005](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-high-parity-delivery-20261005) | `7c23ac8e257fc1ad574244e4accd623c64de4cb8` |
| [paper-studies/20261009/nep-norm-headroom-20261006](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-norm-headroom-20261006) | `be5dcb32fbc443a0fb598d6a6c79b1d71611659d` |
| [paper-studies/20261009/nep-param-compact-20260917](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-param-compact-20260917) | `d638286d785cea312d41e4a06b5b5e0e264e092d` |
| [paper-studies/20261009/nep-param-destination-scatter-20260929](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-param-destination-scatter-20260929) | `e0113b468db47f6f6104bcd92075ec54f2bf5f85` |
| [paper-studies/20261009/nep-param-owner-parallel-20260921](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-param-owner-parallel-20260921) | `f57cc9a65358f557579ebb6dfef8765baa99bb54` |
| [paper-studies/20261009/nep-param-prefetch-20260915](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-param-prefetch-20260915) | `51fdbbb11da3f9aa1a07f1bd52b7e3625151c7a9` |
| [paper-studies/20261009/nep-param-waves-20260922](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-param-waves-20260922) | `8f840ff72d23b9a0d2c4e7bdd9ca690cd6f4a706` |
| [paper-studies/20261009/nep-placement-ablation-20260928](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-placement-ablation-20260928) | `1f084d5dc5acfce415cda906c71af13113a2b6b9` |
| [paper-studies/20261009/nep-placement-controlled-20260929](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-placement-controlled-20260929) | `4142d0bcb0e87eeaac1e02b4f72287c088d1f1a6` |
| [paper-studies/20261009/nep-real-ablation-20261002](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-real-ablation-20261002) | `0cee1d14ad9f3113f43dad6cb47ae80530ffd566` |
| [paper-studies/20261009/nep-release-ablation-20260928](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-release-ablation-20260928) | `3225f09a1ab937ac87b4e4b1592fa75341791b48` |
| [paper-studies/20261009/nep-seq32k-ga2-20261008](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-seq32k-ga2-20261008) | `70877f7d8b4f8f4519c7feaace39414ffd496a66` |
| [paper-studies/20261009/nep-slot-cache-20260917](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-slot-cache-20260917) | `e105bd46c3966d916a6cb0276866eb85530e0529` |
| [paper-studies/20261009/nep-static-slot-capacity-20260921](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-static-slot-capacity-20260921) | `b798119e9d8f7e04b215c708deb958016056f85f` |
| [paper-studies/20261009/nep-sync-cpu-20260914](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-sync-cpu-20260914) | `93d313df4d5588795ea628592668b188ed352a66` |
| [paper-studies/20261009/nep-work-allocation-20260930](https://github.com/daiyaanarfeen/Megatron-LM/tree/paper-studies/20261009/nep-work-allocation-20260930) | `bdcac92326ed99cf98232ac85b2e6bcd8fa6a67f` |

## Archive Verification

Python parsing, shell syntax, and JSON parsing were checked for newly archived local
study source. GPU experiments were not rerun for this documentation/archive operation.
The login node lacks `uv`; benchmark source was preserved without formatting rewrites.
Backup `.orig` files, temporary override scripts, unrelated AGENTS.md edits, and large
generated artifacts are excluded.
