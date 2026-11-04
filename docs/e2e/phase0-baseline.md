# Phase 0 — baseline (fa4, fp8 KV) on H100, Qwen3.8-27B-FP8

Run: `/network-volume/rnd-sglang/runs/green/bench/baseline-fa4/` (launch `runs/fa4fp8/launch.sh`,
`--attention-backend fa4 --kv-cache-dtype fp8_e4m3`, MTP spec-decode, hicache 64 GB write_through,
`--mamba-track-interval 256`), sglang fork `41c0cb3b4c` (branch `main-merge`). Driver: `GreenOptimizer/tools/bench_point.sh`.

## livekit agentic trace, 32 lanes, 1200 s

| metric | value |
|---|---|
| requests | 1508 |
| req/s | 1.23 |
| total tok/s (in+out) | **26,269** |
| TTFT p50 / p90 | 2.90 s / 60.66 s |
| per-user decode tok/s p10 / p50 | 5.4 / 12.1 |
| client-visible cache read | 50.7 % (trace ideal 88.3 %) |

Server counters over the run (`livekit/metrics_end.txt`):

| counter | tokens |
|---|---|
| prompt tokens | 39.97 M |
| cache hit, device | 14.92 M |
| cache hit, host (L2) | 7.11 M |
| prefill computed | 17.94 M |
| evicted from device | 23.71 M |
| backed up to host (KV) | 16.61 M |

Prefill compute ran at ≈14.6 k tok/s average (17.94 M / 1227 s), i.e. the GPU is prefill-bound
recomputing prefixes that were cached and then evicted. Decode tokens are negligible (43 k).

## Memory configuration (server.log)

| pool | size |
|---|---|
| weights | 28.39 GB |
| device KV (fp8) | 117,725 tokens, 3.6 GB (+ draft KV bf16 2.24 GB) |
| device mamba (fp32 SSM) | 155 slots, 22.7 GB → `max_running_requests` capped at 51 |
| host KV | 242,119 tokens, 7.93 GB |
| host mamba | 365 slots, 56.19 GB |
| host draft KV | 242,119 tokens, 4.96 GB |
| GPU free after graph capture | 10.49 GB |

`--hicache-size 64` is split pro-rata by *device* bytes (`hybrid_pool_assembler._split_hicache_size`),
so the fp32 mamba pool takes 88 % of the host budget and host KV holds only ≈11 livekit sessions
(mean ISL 21.5 k). The working set of 32 live sessions is ≈700 k tokens. Device + host KV
(360 k tokens) cannot hold it, so roughly half of all prefix tokens are recomputed.

Root cause of the gap to 2×: KV cache capacity and pool split, not attention-kernel speed.

## Fix: re-split the memory budget (no code change)

Flags appended to the baseline launch: `--mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 1
--hicache-size 100` (run `cache-bf16-r1-h100/`).

| pool | baseline | cache config |
|---|---|---|
| device mamba | 155 slots fp32, 22.7 GB | 150 slots bf16, 11.3 GB (max running 50) |
| device KV | 117,725 tokens | 346,275 tokens |
| host KV | 242,119 tokens, 7.93 GB | 1,312,856 tokens, 43.02 GB |
| host mamba | 365 slots, 56.19 GB | 727 slots, 57.03 GB |
| host draft KV | 4.96 GB | 26.89 GB |

| livekit c32 | baseline | cache config | ratio |
|---|---|---|---|
| req/s | 1.23 | 4.44 | 3.61× |
| total tok/s | 26,269 | **96,142** | **3.66×** |
| TTFT p50 / p90 | 2.90 / 60.66 s | 0.33 / 1.66 s | |
| per-user tok/s p10 / p50 | 5.4 / 12.1 | 12.7 / 58.7 | |
| cache read | 50.7 % | 91.9 % | |
| prefill computed | 17.94 M tokens | 10.75 M tokens (for 3.6× the requests) | |
| GSM8K (1319 q) | 0.942 | 0.945 | |

bf16 SSM state does not cost accuracy on GSM8K (0.945 vs 0.942).
