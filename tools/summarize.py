"""Markdown table of every bench_point.sh result under a root directory.

    python summarize.py <bench_root> [tag ...]

One row per <tag>/<workload>/aiperf_artifacts/profile_export.jsonl (parsed by
green_optimizer.runner.parse_profile_export), plus GSM8K accuracy from <tag>/gsm8k.log.
"""

import re
import sys
from pathlib import Path

from green_optimizer.runner import parse_profile_export


def main(root: str, tags: list[str]) -> None:
    base = Path(root)
    dirs = [base / t for t in tags] if tags else sorted(p for p in base.iterdir() if p.is_dir())
    print("| tag | workload | n | req/s | tok/s | TTFT p50 s | TTFT p90 s | user tok/s p10/p50 | cache hit |")
    print("|---|---|---|---|---|---|---|---|---|")
    accs = []
    for d in dirs:
        for f in sorted(d.glob("*/aiperf_artifacts/profile_export.jsonl")):
            r = parse_profile_export(str(f))
            wl = f.parent.parent.name
            if not r["ok"]:
                print(f"| {d.name} | {wl} | {r['error']} |||||||")
                continue
            print(
                f"| {d.name} | {wl} | {r['n_requests']} | {r['req_s']:.2f} | {r['throughput_tok_s']:,.0f} "
                f"| {r['ttft_p50_ms'] / 1e3:.2f} | {r['ttft_p90_ms'] / 1e3:.2f} "
                f"| {r['tok_s_slow']:.1f}/{r['tok_s_p50']:.1f} | {r['cache_hit_rate']:.1%} |"
            )
        log = d / "gsm8k.log"
        if log.exists():
            m = re.search(r"Accuracy:\s*([0-9.]+)", log.read_text())
            accs.append(f"{d.name}: GSM8K accuracy {m.group(1) if m else 'n/a'}")
    for a in accs:
        print(f"\n{a}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
