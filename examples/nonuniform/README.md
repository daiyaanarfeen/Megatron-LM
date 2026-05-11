# Nonuniform GPT Benchmarks

`pretrain_gpt_nonuniform.py` is a standard GPT pretraining entrypoint with script-local opt-in
DDP wrappers for NTP and NEP. It does not modify Megatron's generic training loop.

These benchmarks intentionally use the non-distributed optimizer design:

- omit `--use-distributed-optimizer`
- enable `--overlap-grad-reduce` when measuring overlap
- let `finish_grad_sync()` return synced gradients to the ranks that own local params before the
  normal optimizer step

Example NTP launch:

```bash
torchrun --nproc-per-node 8 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode tp \
  --nonuniform-tp-base 8 \
  --nonuniform-tp-spares 2 \
  --nonuniform-tp-num-reduced-dp-ranks 1 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

For topology-aware NTP, use `--nonuniform-tp-domain-sizes` to map global ranks
into contiguous physical TP/NVL domains before Megatron model-parallel groups are
created:

```bash
torchrun --nproc-per-node 4 --nnodes 3 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode tp \
  --tensor-model-parallel-size 4 \
  --context-parallel-size 1 \
  --nonuniform-tp-base 4 \
  --nonuniform-tp-spares 2 \
  --nonuniform-tp-domain-sizes 2 4 4 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

Here the first replica uses TP2 and the next two replicas use TP4; each TP group
is built inside its contiguous rank block. The values must currently be either
`tp_base` or `tp_base - tp_spares`, which preserves the existing reduced/full TP
resharding semantics.

Example NEP launch:

```bash
torchrun --nproc-per-node 16 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode ep \
  --expert-model-parallel-size 16 \
  --nonuniform-ep-min-size 12 \
  --nonuniform-ep-placement-path /path/to/ep_placement.json \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

For full nonuniform EP topology setup, prefer the topology flag:

```bash
torchrun --nproc-per-node 4 --nnodes 15 examples/nonuniform/pretrain_gpt_nonuniform.py \
  --nonuniform-mode ep \
  --tensor-model-parallel-size 2 \
  --context-parallel-size 2 \
  --expert-tensor-parallel-size 1 \
  --nonuniform-ep-num-tp-cp-per-replica 8 7 \
  --num-experts 224 \
  --overlap-grad-reduce \
  ...standard pretrain_gpt.py arguments...
```

With `TP2 CP2 ETP1`, `8 7` creates EP32/EP28 expert replicas. The topology path
creates the nonuniform expert process groups, generates an interleaved placement,
registers that placement before model construction, and uses the same runtime config
for token routing and NEP DDP gradient ownership transfer.

HSG benchmark launcher:

```bash
ssh hsg-1
cd /lustre/fsw/portfolios/coreai/users/darfeen/Megatron-LM-nep-ntp-shared-port
sbatch scripts/nonuniform/run_hsg_ep32_28_tp2cp2_compare.sh
```

The launcher requests `--segment=16` so the 16-node uniform run and the 15-node
nonuniform run are allocated inside one HSG segment. It compares standard
`pretrain_gpt.py` uniform EP32 against `pretrain_gpt_nonuniform.py` with the
script-local NEP opt-in wrapper for EP32/EP28. Defaults are configurable through
environment variables such as `TRAIN_ITERS`, `SEQ_LENGTH`, `NUM_LAYERS`,
`UNIFORM_GBS`, and `NONUNIFORM_GBS`.

The manual NEP placement file is a JSON list with one entry per EP rank. Each entry lists the
global expert IDs physically present on that EP rank in ascending order. Optional owner overrides
use `--nonuniform-ep-expert-owner-path` with a JSON object mapping expert ID to owner EP rank.
