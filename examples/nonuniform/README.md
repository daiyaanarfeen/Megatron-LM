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

The NEP placement file is a JSON list with one entry per EP rank. Each entry lists the global
expert IDs physically present on that EP rank in ascending order. Optional owner overrides use
`--nonuniform-ep-expert-owner-path` with a JSON object mapping expert ID to owner EP rank.
The same placement table is registered before model construction and is used by the MoE layer
and token dispatcher, so forward token routing sends each expert's tokens to its physical holder
inside the local EP group before the NEP DDP wrapper handles gradient ownership transfer.
