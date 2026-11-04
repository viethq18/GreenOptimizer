"""GPU time breakdown of a torch-profiler trace (sglang /start_profile output).

Usage: prof_breakdown.py <trace.json[.gz] | dir> [--top N]

Prints, over the GPU kernel/memcpy timeline of the trace:
  * span, busy (union of device intervals) and idle fraction, plus the idle-gap histogram;
  * time per kernel group (GEMM, attention, GDN, ...) and the top-N kernels by total time.
Groups are name-regex based; unmatched kernels land in "other" and show up in the top-N list.
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = [
    ("green_attn", r"attn_decode_fp8|green"),
    ("gdn_recurrent", r"gdn_recurrent|fused_recurrent|recurrent_gated|sigmoid_gating_delta"),
    ("gdn_chunk", r"chunk_gated_delta|chunk_fwd|chunk_scaled_dot|chunk_local_cumsum|solve_tril|recompute_w_u|chunk_.*_kernel|l2norm"),
    ("conv1d", r"causal_conv1d|conv1d"),
    ("attention", r"flash|fmha|FlashAttn|attention|mha_|attn"),
    ("gemm", r"gemm|cutlass|xmma|nvjet|deep_gemm|matmul|sm90_|wgmma|cublas|w8a8|fp8_scaled_mm|Gemm"),
    ("quant", r"quant|per_token|scaled_fp8|fp8_cast"),
    ("norm_act", r"rms|norm|silu|act_and_mul|gelu|swiglu|sigmoid"),
    ("rope", r"rope|rotary"),
    ("sampling_spec", r"sample|argmax|topk|top_k|softmax|verify|accept|tree|dflash|spec"),
    ("kv_mem", r"store_kv|set_kv|kv_buffer|copy_|transfer|hicache|mamba|state"),
    ("elementwise", r"elementwise|vectorized|unrolled|reduce|Fill|fill|cat|index|scatter|gather|where|copy"),
]
GROUP_RE = [(g, re.compile(p, re.IGNORECASE)) for g, p in GROUPS]


def group_of(name: str) -> str:
    if name.startswith("Memcpy") or name.startswith("Memset"):
        return "memcpy/memset"
    for g, rx in GROUP_RE:
        if rx.search(name):
            return g
    return "other"


def load(path: Path) -> list:
    if path.is_dir():
        files = sorted(path.rglob("*.trace.json*"))
        if not files:
            sys.exit(f"no *.trace.json[.gz] under {path}")
        path = files[0]
        print(f"trace: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    evs = [
        e
        for e in load(a.trace)
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not evs:
        sys.exit("no GPU events in trace")
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs)
    t0, t1 = iv[0][0], max(e for _, e in iv)
    span = t1 - t0
    busy, gaps = 0.0, []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs
            gaps.append(s - ce)
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs

    print(f"GPU events {len(evs)}  span {span / 1e3:.1f} ms  busy {busy / 1e3:.1f} ms "
          f"({100 * busy / span:.1f} %)  idle {(span - busy) / 1e3:.1f} ms")
    edges = [5, 20, 100, 1000, 10000, float("inf")]
    lo = 0
    for hi in edges:
        sel = [g for g in gaps if lo <= g < hi]
        print(f"  idle gaps {lo:>6}-{hi:<6} us: n={len(sel):6d}  total {sum(sel) / 1e3:8.1f} ms")
        lo = hi

    by_group = defaultdict(float)
    by_name = defaultdict(lambda: [0.0, 0])
    for e in evs:
        by_group[group_of(e["name"])] += e["dur"]
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
    tot = sum(by_group.values())
    print(f"\n| group | ms | % of kernel time |\n|---|---|---|")
    for g, t in sorted(by_group.items(), key=lambda x: -x[1]):
        print(f"| {g} | {t / 1e3:.1f} | {100 * t / tot:.1f} |")
    print(f"\n| kernel | group | n | total ms | avg us |\n|---|---|---|---|---|")
    for n, (t, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[: a.top]:
        print(f"| {n[:110]} | {group_of(n)} | {c} | {t / 1e3:.1f} | {t / c:.1f} |")


if __name__ == "__main__":
    main()
