"""Greedy parity probe: run N fixed prompts one at a time (batch 1, deterministic) and save outputs.

    python parity.py run  <url> <out.json> [--n 32] [--max-new-tokens 64]
    python parity.py diff <a.json> <b.json>

`diff` reports exact-match count and, per mismatch, the first differing character offset.
"""

import argparse
import json
import sys
import urllib.request

TOPICS = [
    "the history of the printing press", "how a transistor works", "the water cycle",
    "why the sky is blue", "the rules of chess", "photosynthesis", "the French revolution",
    "how vaccines train the immune system", "plate tectonics", "the causes of inflation",
    "how a compiler optimizes loops", "the life of Marie Curie", "black holes",
    "the Pythagorean theorem", "how TCP congestion control works", "the Silk Road",
    "the structure of DNA", "how airplanes generate lift", "the Roman aqueducts",
    "the theory of evolution", "how a hash table handles collisions", "the Great Barrier Reef",
    "the invention of the telephone", "how GPS determines position", "the Renaissance",
    "how batteries store energy", "the Fibonacci sequence", "the Apollo 11 mission",
    "how neural networks learn", "the Amazon rainforest", "the Industrial Revolution",
    "how public-key cryptography works",
]


def prompts(n: int) -> list[str]:
    out = []
    for i in range(n):
        t = TOPICS[i % len(TOPICS)]
        out.append(
            "<|im_start|>user\nExplain " + t + " in a few clear sentences.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    return out


def run(url: str, out: str, n: int, max_new: int) -> None:
    res = []
    for p in prompts(n):
        body = json.dumps(
            {"text": p, "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
        ).encode()
        req = urllib.request.Request(url.rstrip("/") + "/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            res.append(json.loads(r.read())["text"])
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {len(res)} outputs to {out}")


def diff(a: str, b: str) -> int:
    xa, xb = json.load(open(a)), json.load(open(b))
    same = 0
    for i, (s, t) in enumerate(zip(xa, xb)):
        if s == t:
            same += 1
            continue
        k = next((j for j, (c, d) in enumerate(zip(s, t)) if c != d), min(len(s), len(t)))
        print(f"[{i}] differs at char {k}/{len(s)}: {s[max(0, k - 20):k + 30]!r} vs {t[max(0, k - 20):k + 30]!r}")
    print(f"exact match {same}/{len(xa)}")
    return 0 if same == len(xa) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("out")
    r.add_argument("--n", type=int, default=32)
    r.add_argument("--max-new-tokens", type=int, default=64)
    d = sub.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.url, a.out, a.n, a.max_new_tokens)
    else:
        sys.exit(diff(a.a, a.b))
