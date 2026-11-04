#!/usr/bin/env python3
"""Group the CUDA time of SGLang torch-profiler traces (``/start_profile``/``/stop_profile``,
``SGLANG_TORCH_PROFILER_DIR``) into the Phase 0 buckets:

  full-attn decode | full-attn verify | full-attn prefill |
  GDN decode | GDN verify | GDN extend | GEMM | norm/rope/gate/other

Stage comes from the trace: SGLang writes one trace per stage when profiling with
``start_step``/``num_steps`` and stage split, named ``*-EXTEND*`` / ``*-DECODE*``
(a file without either marker uses ``--stage``). With ``--spec`` the decode-stage attention/GDN
kernels are verify kernels (DFLASH verify runs as a q_len>1 extend-style call inside the decode
loop); without it they are plain decode.

Only ``cat == "kernel"`` events count (memcpy/memset/runtime are "other").
Usage: prof_breakdown.py TRACE_OR_DIR [...] [--stage decode|extend|auto] [--spec] [--json OUT]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

BUCKETS = (
    "full-attn decode",
    "full-attn verify",
    "full-attn prefill",
    "GDN decode",
    "GDN verify",
    "GDN extend",
    "GEMM",
    "norm/rope/gate/other",
)

# Order matters: GDN is tested before attention (GDN kernel names contain "attention"-free
# words but conv/delta names must win), attention before GEMM (CuTe-DSL FA kernels carry
# "cutlass" in their names).
_GDN_EXTEND = re.compile(
    r"chunk_gated_delta_rule|chunk_delta_h|chunk_local_cumsum|recompute_w_u|chunk_scaled_dot_kkt|"
    r"fwd_kkt_solve|gdn_prefill|causal_conv1d_fwd|l2norm_fwd_kernel|solve_tril|chunk_fwd_o|"
    r"flashinfer.*(gdn|delta)|gated_delta.*prefill|merge_16x16",
    re.I,
)
_GDN_DECODE = re.compile(
    r"fused_recurrent_gated_delta_rule|causal_conv1d_update|gdn_kernel_(small|large)_batch|replayssm|"
    r"gdn|delta_rule|green_gdn",
    re.I,
)
_ATTN = re.compile(
    r"FlashAttentionForward|flash_fwd|FlashAttnFwd|FlashAttentionForwardCombine|flash_attn|"
    r"BatchPrefill|BatchDecode|PagedAttention|fmha|green_attn|attn_decode|attn_verify|"
    r"mha_|splitkv|split_kv|flash_decoding|_attn_fwd|softmax_lse|"
    r"flash::|flash_",
    re.I,
)
_GEMM = re.compile(
    r"nvjet|cutlass|gemm|cublas|sm90_xmma|sm80_xmma|xmma|deep_gemm|fp8_gemm|splitk|gemv|wgmma|cutlass3x|ampere_|hopper_",
    re.I,
)


def classify(name: str, stage: str, spec: bool) -> str:
    """Bucket of kernel ``name`` when the trace stage is ``stage`` ('decode' or 'extend')."""
    if _GDN_EXTEND.search(name):
        return "GDN extend" if stage == "extend" else ("GDN verify" if spec else "GDN decode")
    if _GDN_DECODE.search(name):
        # gdn/delta-rule recurrent kernels: extend stage runs the chunked path (FlashInfer)
        if stage == "extend":
            return "GDN extend"
        return "GDN verify" if spec else "GDN decode"
    if _ATTN.search(name):
        if stage == "extend":
            return "full-attn prefill"
        return "full-attn verify" if spec else "full-attn decode"
    if _GEMM.search(name):
        return "GEMM"
    return "norm/rope/gate/other"


def trace_stage(path: Path, default: str) -> str:
    n = path.name.upper()
    if "EXTEND" in n or "PREFILL" in n:
        return "extend"
    if "DECODE" in n:
        return "decode"
    return default


def _load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def breakdown(trace: dict, stage: str, spec: bool) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """(bucket -> microseconds, bucket -> {kernel name -> microseconds})"""
    total: Dict[str, float] = defaultdict(float)
    names: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        name = ev.get("name", "")
        if ev.get("cat") == "kernel":
            b = classify(name, stage, spec)
        elif ev.get("cat") in ("gpu_memcpy", "gpu_memset"):
            b = "norm/rope/gate/other"
        else:
            continue
        total[b] += ev["dur"]
        names[b][name] += ev["dur"]
    return dict(total), {k: dict(v) for k, v in names.items()}


def collect(paths: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(list(p.rglob("*.json")) + list(p.rglob("*.json.gz")))
        else:
            out.append(p)
    return out


def report(total: Dict[str, float], names: Dict[str, Dict[str, float]], top: int = 3) -> str:
    grand = sum(total.values()) or 1.0
    lines = [f"{'bucket':<24}{'ms':>12}{'share':>9}  top kernels"]
    for b in BUCKETS:
        us = total.get(b, 0.0)
        tops = sorted(names.get(b, {}).items(), key=lambda kv: -kv[1])[:top]
        lines.append(f"{b:<24}{us / 1e3:>12.2f}{us / grand:>8.1%}  " + "; ".join(f"{n[:48]} {t / 1e3:.1f}ms" for n, t in tops))
    lines.append(f"{'total':<24}{grand / 1e3:>12.2f}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace .json/.json.gz files or directories")
    ap.add_argument("--stage", choices=("decode", "extend"), default="decode", help="stage of traces whose name carries no EXTEND/DECODE marker")
    ap.add_argument("--spec", action="store_true", help="speculative decoding on: decode-stage attention/GDN count as verify")
    ap.add_argument("--json", help="also write the bucket totals (us) here")
    a = ap.parse_args(argv)
    files = collect(a.traces)
    if not files:
        print("no trace files found", file=sys.stderr)
        return 2
    total: Dict[str, float] = defaultdict(float)
    names: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for f in files:
        t, n = breakdown(_load(f), trace_stage(f, a.stage), a.spec)
        for b, us in t.items():
            total[b] += us
            for k, v in n[b].items():
                names[b][k] += v
    print(report(total, names))
    if a.json:
        Path(a.json).write_text(json.dumps({"files": [str(f) for f in files], "us": dict(total)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
