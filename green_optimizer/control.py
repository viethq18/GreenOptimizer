"""Online sidecar (O2): poll /metrics, steer along a ladder through /set_internal_state.

Pure-python (stdlib + pyyaml). Policy (plan O2), windows are ``window_s`` wide, polled every
``interval_s``:
  * TTFT p90 over SLO for 2 consecutive evaluations  -> one rung toward TTFT priority
  * slow-tail tok/s under SLO for 2 consecutive evals -> one rung toward throughput priority
  * both violated -> hold, log "infeasible at load"
Histograms are cumulative in Prometheus, so a sliding window is the bucket delta between the
snapshot now and the snapshot ``window_s`` ago.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import yaml

from .spec import SLO

log = logging.getLogger("green_optimizer.control")

TTFT_METRIC = "sglang:time_to_first_token_seconds"
ITL_METRIC = "sglang:inter_token_latency_seconds"

Buckets = Dict[float, float]  # upper bound (le) -> cumulative count

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_histogram(text: str, name: str, label_filter: Optional[Dict[str, str]] = None) -> Buckets:
    """Cumulative bucket counts of histogram ``name``, summed over every label set matching
    ``label_filter`` (extra labels such as engine_type/model_name are summed over)."""
    out: Buckets = {}
    want = f"{name}_bucket"
    for line in text.splitlines():
        if not line or line[0] == "#" or not line.startswith(want):
            continue
        m = _SAMPLE.match(line.strip())
        if not m or m.group(1) != want:
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        le = labels.pop("le", None)
        if le is None:
            continue
        if label_filter and any(labels.get(k) != v for k, v in label_filter.items()):
            continue
        ub = math.inf if le in ("+Inf", "Inf") else float(le)
        out[ub] = out.get(ub, 0.0) + float(m.group(3))
    return out


def bucket_delta(new: Buckets, old: Optional[Buckets]) -> Buckets:
    if not old:
        return dict(new)
    d = {ub: new[ub] - old.get(ub, 0.0) for ub in new}
    if any(v < 0 for v in d.values()):  # server restarted: counters reset
        return dict(new)
    return d


def histogram_quantile(buckets: Buckets, q: float) -> Optional[float]:
    """Prometheus-style linear interpolation inside the bucket holding the q-quantile.
    None when the window holds no observations. The +Inf bucket resolves to the largest
    finite bound."""
    if not buckets:
        return None
    items = sorted(buckets.items())
    total = items[-1][1]
    if total <= 0:
        return None
    rank = q * total
    prev_ub, prev_c = 0.0, 0.0
    for ub, c in items:
        if c >= rank:
            if math.isinf(ub):
                return prev_ub
            if c == prev_c:
                return ub
            return prev_ub + (ub - prev_ub) * (rank - prev_c) / (c - prev_c)
        prev_ub, prev_c = ub, c
    return prev_ub


def histogram_count(buckets: Buckets) -> float:
    return max(buckets.values()) if buckets else 0.0


# --------------------------------------------------------------------------- ladder
@dataclass
class Rung:
    rung: int
    online: Dict[str, int]  # chunked_prefill_size / max_running_requests
    ttft_p90_ms: float = 0.0
    tok_s_slow: float = 0.0
    source: str = ""


def load_ladder(path: str) -> List[Rung]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    rows = doc["ladder"] if isinstance(doc, dict) else doc
    rungs = [
        Rung(
            rung=int(r.get("rung", i)),
            online=dict(r["online"]),
            ttft_p90_ms=float(r.get("ttft_p90_ms", 0.0)),
            tok_s_slow=float(r.get("tok_s_slow", 0.0)),
            source=str(r.get("source", "")),
        )
        for i, r in enumerate(rows)
    ]
    if not rungs:
        raise ValueError(f"empty ladder: {path}")
    return rungs


def server_args_for(rung: Rung) -> Dict[str, int]:
    """The /set_internal_state payload keys (plugin handlers, see plugin.py)."""
    return {
        "green_chunked_prefill_size": int(rung.online["chunked_prefill_size"]),
        "green_max_running_cap": int(rung.online["max_running_requests"]),
    }


# --------------------------------------------------------------------------- policy
@dataclass
class Decision:
    action: str  # "hold" | "to_ttft" | "to_throughput" | "infeasible"
    rung: int  # target rung index
    ttft_violated: bool = False
    tok_violated: bool = False


class LadderPolicy:
    """Discrete hysteresis controller. Rung 0 is TTFT-priority, the last rung throughput-priority.

    ``patience`` consecutive violating evaluations are required before moving; a move resets
    both counters (the new config needs a fresh window to prove itself)."""

    def __init__(self, n_rungs: int, start: int, slo: SLO, patience: int = 2):
        if not 0 <= start < n_rungs:
            raise ValueError("start rung out of range")
        self.n = n_rungs
        self.rung = start
        self.slo = slo
        self.patience = patience
        self._ttft_streak = 0
        self._tok_streak = 0

    def step(self, ttft_p90_ms: Optional[float], tok_s_slow: Optional[float]) -> Decision:
        tv = ttft_p90_ms is not None and ttft_p90_ms > self.slo.ttft_p90_ms
        kv = tok_s_slow is not None and tok_s_slow < self.slo.user_tok_s_p90_min
        if tv and kv:
            # contradictory demands: moving either way trades one violation for the other
            self._ttft_streak = self._tok_streak = 0
            return Decision("infeasible", self.rung, tv, kv)
        self._ttft_streak = self._ttft_streak + 1 if tv else 0
        self._tok_streak = self._tok_streak + 1 if kv else 0
        if self._ttft_streak >= self.patience:
            self._ttft_streak = self._tok_streak = 0
            if self.rung > 0:
                self.rung -= 1
                return Decision("to_ttft", self.rung, tv, kv)
            return Decision("hold", self.rung, tv, kv)
        if self._tok_streak >= self.patience:
            self._ttft_streak = self._tok_streak = 0
            if self.rung < self.n - 1:
                self.rung += 1
                return Decision("to_throughput", self.rung, tv, kv)
            return Decision("hold", self.rung, tv, kv)
        return Decision("hold", self.rung, tv, kv)


# --------------------------------------------------------------------------- sidecar
class WindowStats:
    """Keeps (t, ttft_buckets, itl_buckets) snapshots and answers window quantiles."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._snaps: Deque[Tuple[float, Buckets, Buckets]] = deque()

    def add(self, t: float, ttft: Buckets, itl: Buckets) -> None:
        self._snaps.append((t, ttft, itl))
        # keep exactly one snapshot at or before the window start
        while len(self._snaps) > 2 and self._snaps[1][0] <= t - self.window_s:
            self._snaps.popleft()

    def reset(self) -> None:
        """Forget history (after a config change the old window mixes two configs)."""
        if self._snaps:
            self._snaps = deque([self._snaps[-1]])

    def metrics(self, min_samples: int = 5) -> Tuple[Optional[float], Optional[float], float]:
        """(ttft_p90_ms, tok_s_slow, n_requests) over the window.

        tok_s_slow = 1 / ITL_p90: the slowest decile of inter-token gaps. With speculative
        decoding ITL is per emitted token, so this approximates the per-user rate; it is
        an approximation of aiperf's per-request p10 and is documented as such in the report."""
        if len(self._snaps) < 2:
            return None, None, 0.0
        t0, ttft0, itl0 = self._snaps[0]
        t1, ttft1, itl1 = self._snaps[-1]
        dt = bucket_delta(ttft1, ttft0)
        di = bucket_delta(itl1, itl0)
        n = histogram_count(dt)
        ttft = histogram_quantile(dt, 0.9) if n >= min_samples else None
        itl = histogram_quantile(di, 0.9) if histogram_count(di) >= min_samples else None
        return (
            ttft * 1000.0 if ttft is not None else None,
            1.0 / itl if itl else None,
            n,
        )


def http_get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def http_post_json(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class ApplyError(RuntimeError):
    pass


def apply_rung(url: str, rung: Rung, post: Callable[..., Dict[str, Any]] = http_post_json) -> Dict[str, Any]:
    """POST the rung; raise ApplyError unless the server answers updated=true."""
    resp = post(f"{url.rstrip('/')}/set_internal_state", {"server_args": server_args_for(rung)})
    if not resp.get("updated", False):
        raise ApplyError(f"set_internal_state rejected rung {rung.rung}: {resp}")
    return resp


@dataclass
class Controller:
    url: str
    ladder: List[Rung]
    slo: SLO
    interval_s: float = 5.0
    window_s: float = 60.0
    start_rung: Optional[int] = None
    get: Callable[[str], str] = http_get
    post: Callable[..., Dict[str, Any]] = http_post_json
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # default start: the most throughput-oriented rung that still meets the SLO per the
        # ladder (rungs are produced SLO-satisfying first), i.e. the last rung
        start = len(self.ladder) - 1 if self.start_rung is None else self.start_rung
        self.policy = LadderPolicy(len(self.ladder), start, self.slo)
        self.stats = WindowStats(self.window_s)

    def _emit(self, **kw) -> None:
        self.events.append({"t": self.clock(), **kw})
        log.info("%s", kw)

    def start(self) -> None:
        apply_rung(self.url, self.ladder[self.policy.rung], self.post)
        self._emit(event="apply", rung=self.policy.rung, why="start")

    def tick(self) -> Decision:
        text = self.get(f"{self.url.rstrip('/')}/metrics")
        ttft = parse_histogram(text, TTFT_METRIC, {"is_streaming": "true"}) or parse_histogram(text, TTFT_METRIC)
        itl = parse_histogram(text, ITL_METRIC)
        self.stats.add(self.clock(), ttft, itl)
        p90_ms, tok_slow, n = self.stats.metrics()
        d = self.policy.step(p90_ms, tok_slow)
        self._emit(event="window", ttft_p90_ms=p90_ms, tok_s_slow=tok_slow, n=n, decision=d.action, rung=d.rung)
        if d.action == "infeasible":
            log.warning("infeasible at load: ttft_p90=%s ms and tok/s=%s both violate the SLO", p90_ms, tok_slow)
        elif d.action in ("to_ttft", "to_throughput"):
            apply_rung(self.url, self.ladder[d.rung], self.post)
            self.stats.reset()
            self._emit(event="apply", rung=d.rung, why=d.action)
        return d

    def run(self, duration_s: Optional[float] = None) -> None:
        self.start()
        t_end = None if duration_s is None else self.clock() + duration_s
        while t_end is None or self.clock() < t_end:
            self.sleep(self.interval_s)
            self.tick()
