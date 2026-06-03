# NTP Mamba HSG Benchmark

This runner compares a uniform TP Mamba baseline against the merged opt-in NTP
implementation from `upstream/dev` PR 4585. It keeps all generated source copies,
logs, summaries, and PyTorch chrome traces under `hsg_profile_traces/ntp_mamba`.

Submit from this repository on an HSG login node:

```bash
sbatch tools/hsg_ntp_mamba_benchmark.sbatch
```

Defaults:

- uniform: TP4 x DP2 on 8 GPUs, micro batch 2.
- NTP: packed TP2 + TP4 on 6 GPUs, reduced replica micro batch 1 and healthy
  replica micro batch 2.
- model: all-Mamba `MambaModel`, 8 layers, hidden 1024, seq 1024, bf16.
- traces: uniform ranks `0,4`; NTP ranks `0,2,4,5`.

Useful overrides:

```bash
LAYERS=12 HIDDEN_SIZE=2048 SEQ_LEN=2048 STEPS=12 sbatch tools/hsg_ntp_mamba_benchmark.sbatch
```

Watch for up to five minutes:

```bash
job=<jobid>
deadline=$((SECONDS + 300))
while (( SECONDS < deadline )); do
  squeue -j "$job" -o "%.18i %.9T %.10M %.20S %.40R"
  sleep 30
done
squeue -j "$job" -o "%.18i %.9T %.10M %.20S %.40R"
```
