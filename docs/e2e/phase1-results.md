# Phase 1 — GreenAttention E2E + config levers (H100, Qwen3.8-27B-FP8, livekit c32)

All runs: `/network-volume/rnd-sglang/runs/green/bench/<tag>/`, driver
`GreenOptimizer/tools/bench_point.sh`, table via `tools/summarize.py`. Baseline and memory
analysis: `phase0-baseline.md`.

`C` = `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 1 --hicache-size 100`.
Green = `SGLANG_PLUGINS=green_attention --attention-backend green
--speculative-draft-attention-backend fa4` (GreenAttention `f40ad7f`).

## livekit c32 (1200 s)

| tag | config | req/s | tok/s | vs baseline | TTFT p50 / p90 s | user tok/s p10 / p50 | cache hit |
|---|---|---|---|---|---|---|---|
| baseline-fa4 | launch.sh | 1.23 | 26,269 | 1.00× | 2.90 / 60.66 | 5.4 / 12.1 | 50.7 % |
| cache-bf16-r1-h100 | fa4 + C | 4.44 | 96,142 | 3.66× | 0.33 / 1.66 | 12.7 / 58.7 | 91.9 % |
| green-cache | Green + C | 4.55 | 98,619 | 3.75× | 0.29 / 1.59 | 15.4 / 66.9 | 91.8 % |
| green-mf088 | Green + C + `--mem-fraction-static 0.88` | 4.56 | 98,832 | 3.76× | 0.29 / 1.59 | 15.2 / 66.6 | 91.8 % |

Green over fa4 at equal config: +2.6 % total tok/s and +14 % median / +21 % p10 per-user decode
speed. Total throughput moves less than decode speed because the lane cycle is dominated by trace
think time once the cache works. Mean request latency in `cache-bf16-r1-h100` is 1.56 s
(TTFT 0.65 s + decode ≈0.9 s) out of a 7.2 s cycle, so zero server time would give only
32 / 5.6 s ≈ 5.7 req/s ≈ 123 k tok/s ≈ **4.7× baseline**: the structural ceiling of this workload at
c32. Further gains can only come from server time (TTFT + decode), so the remaining levers are
judged by how far they close the 98.6 k → 123 k gap and by per-user speed.

`green-mf088`: 370,185 KV tokens (+7 %) and max running 55 change nothing (+0.2 %). Device KV
capacity is no longer a limiter at c32.

Kernel dispatch in the C config (server.log): attention verify (q ≤ 10) runs Green, while
prefill-shaped verify (q = 12, 16) falls back. GDN verify falls back because the bf16 SSM state
is outside the V4 kernel's fp32 regime.

## Accuracy

| run | GSM8K (1319 q, 5-shot) |
|---|---|
| baseline fa4 (fp32 state) | 0.942 |
| fa4 + C (bf16 state) | 0.945 |
