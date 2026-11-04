"""Offline configuration search: roofline prefilter -> successive halving -> Pareto ladder.

Round structure (plan O1):
  prefilter   roofline over the whole space; drop configs predicted to violate the SLO by
              more than ``slack`` (20%)
  round 1     top-K by predicted throughput, 200 requests each
  round 2     full bench for the top 25% of round 1
  knob sweep  ``UNMODELLED`` knobs (not in the roofline) varied one at a time around the
              round-2 winner, full bench
  round 3     top 3, repeated 3 times, median of the repeats
The measurement function is injected (``Measure``) so the logic is unit-testable without a GPU.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .roofline import CalibParams, Prediction, predict
from .spec import (
    ONLINE_KNOBS,
    SLO,
    Hardware,
    ModelSpec,
    SearchSpace,
    Workload,
    config_key,
)

# knobs the roofline does not model: their effect is decided by measurement only
UNMODELLED = ("max_prefill_tokens", "schedule_policy")
# default for each unmodelled knob while ranking (the production launch.sh values)
UNMODELLED_DEFAULTS = {"max_prefill_tokens": 16384, "schedule_policy": "dfs-weight"}
SLACK = 0.20
ROUND1_K = 24
ROUND1_REQUESTS = 200
ROUND3_REPEATS = 3

Measurement = Dict[str, Any]  # ttft_p90_ms, tok_s_slow, tok_s_fast, throughput_tok_s, ok, ...
Measure = Callable[[Dict[str, Any], str, int], Measurement]  # (cfg, round_name, repeat) -> m


@dataclass
class Candidate:
    cfg: Dict[str, Any]
    pred: Prediction
    violation: float  # SLO violation of the prediction (0 = met)

    @property
    def key(self) -> str:
        return config_key(self.cfg)


def modelled_key(cfg: Dict[str, Any]) -> str:
    return config_key({k: v for k, v in cfg.items() if k not in UNMODELLED})


def _violation(slo: SLO, pred: Prediction) -> float:
    if not pred.feasible:
        return math.inf
    return slo.violation(pred.ttft_p90_ms, pred.tok_s_slow)


def prefilter(
    space: SearchSpace,
    wl: Workload,
    slo: SLO,
    hw: Hardware,
    model: ModelSpec,
    params: CalibParams,
) -> List[Candidate]:
    """Predict every distinct modelled config; sorted best first.

    Order: configs within SLO (+slack) first by predicted throughput, then the rest by
    increasing violation (so the head of the list is also the best-effort frontier).
    """
    seen = set()
    out: List[Candidate] = []
    for cfg in space.configs():
        cfg = {**cfg, **UNMODELLED_DEFAULTS}
        if config_key(cfg) in seen:
            continue
        seen.add(config_key(cfg))
        pred = predict(cfg, wl, hw, model, params)
        out.append(Candidate(cfg, pred, _violation(slo, pred)))
    out.sort(key=lambda c: _rank_key(c))
    return out


def _rank_key(c: Candidate) -> Tuple[int, float]:
    if not c.pred.feasible:
        return (2, math.inf)
    if c.violation <= SLACK:
        return (0, -c.pred.total_tok_s)
    return (1, c.violation)


def passing(cands: Sequence[Candidate]) -> List[Candidate]:
    return [c for c in cands if c.pred.feasible and c.violation <= SLACK]


# --------------------------------------------------------------------------- halving
def score_key(slo: SLO, m: Measurement) -> Tuple[int, float]:
    """Sort key of a measurement (smaller is better): SLO-met by throughput, else by violation."""
    if not m.get("ok", False):
        return (2, math.inf)
    v = slo.violation(m["ttft_p90_ms"], m["tok_s_slow"])
    if v <= 0:
        return (0, -m["throughput_tok_s"])
    return (1, v)


def median_measurement(ms: Sequence[Measurement]) -> Measurement:
    good = [m for m in ms if m.get("ok", False)]
    if not good:
        return dict(ms[0]) if ms else {"ok": False, "error": "no measurements"}
    out: Dict[str, Any] = {"ok": True, "repeats": len(good)}
    for k in good[0]:
        vals = [m[k] for m in good if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
        if len(vals) == len(good):
            out[k] = statistics.median(vals)
    return out


@dataclass
class SearchResult:
    rounds: Dict[str, List[Tuple[Dict[str, Any], Measurement]]] = field(default_factory=dict)
    winner: Optional[Dict[str, Any]] = None
    winner_measurement: Optional[Measurement] = None
    slo_met: bool = False


def _rank(slo: SLO, rows: List[Tuple[Dict[str, Any], Measurement]]) -> List[Tuple[Dict[str, Any], Measurement]]:
    return sorted(rows, key=lambda r: score_key(slo, r[1]))


def unmodelled_variants(cfg: Dict[str, Any], space: SearchSpace) -> List[Dict[str, Any]]:
    out = []
    for k in UNMODELLED:
        for v in space.space.get(k, []):
            if v != cfg[k]:
                out.append({**cfg, k: v})
    return out


def successive_halving(
    cands: Sequence[Candidate],
    slo: SLO,
    space: SearchSpace,
    measure: Measure,
    k: int = ROUND1_K,
    keep_fraction: float = 0.25,
    top_final: int = 3,
    repeats: int = ROUND3_REPEATS,
    on_result: Optional[Callable[[str, Dict[str, Any], Measurement], None]] = None,
) -> SearchResult:
    res = SearchResult()

    def run(round_name: str, cfgs: Iterable[Dict[str, Any]], repeat: int = 0):
        rows = []
        for cfg in cfgs:
            m = measure(cfg, round_name, repeat)
            if on_result:
                on_result(round_name, cfg, m)
            rows.append((cfg, m))
        res.rounds.setdefault(round_name, []).extend(rows)
        return rows

    r1 = _rank(slo, run("round1", [c.cfg for c in cands[:k]]))
    n2 = max(top_final, math.ceil(len(r1) * keep_fraction))
    r2 = _rank(slo, run("round2", [cfg for cfg, m in r1[:n2] if m.get("ok", False)]))
    if not r2 or not r2[0][1].get("ok", False):
        return res
    sweep = run("knob_sweep", unmodelled_variants(r2[0][0], space))
    pool = _rank(slo, r2 + sweep)
    finalists = [cfg for cfg, m in pool[:top_final] if m.get("ok", False)]
    final_rows = []
    for cfg in finalists:
        reps = [measure(cfg, "round3", i) for i in range(repeats)]
        for i, m in enumerate(reps):
            if on_result:
                on_result("round3", cfg, {**m, "repeat": i})
        final_rows.append((cfg, median_measurement(reps)))
    res.rounds["round3"] = final_rows
    final_rows = _rank(slo, final_rows)
    if final_rows and final_rows[0][1].get("ok", False):
        res.winner, res.winner_measurement = final_rows[0]
        res.slo_met = score_key(slo, res.winner_measurement)[0] == 0
    return res


# --------------------------------------------------------------------------- pareto / ladder
def pareto_front(points: Sequence[Dict[str, Any]], minimise: str, maximise: str) -> List[Dict[str, Any]]:
    """Non-dominated points for (minimise ``minimise``, maximise ``maximise``), sorted by
    ascending ``minimise``. Exact duplicates are collapsed."""
    pts = sorted(points, key=lambda p: (p[minimise], -p[maximise]))
    front: List[Dict[str, Any]] = []
    best = -math.inf
    for p in pts:
        if p[maximise] > best:
            front.append(p)
            best = p[maximise]
    return front


def online_variants(base: Dict[str, Any], space: SearchSpace, chunk_grid=(1024, 2048, 4096, 8192, 16384)) -> List[Dict[str, Any]]:
    """Configs differing from ``base`` only in online knobs (O2 can reach them at runtime):
    chunked_prefill_size in [1024, startup], max_running_requests cap in [1, startup]."""
    chunks = sorted({c for c in (*chunk_grid, base["chunked_prefill_size"]) if 1024 <= c <= base["chunked_prefill_size"]})
    caps = sorted({c for c in (*space.space.get("max_running_requests", []), base["max_running_requests"]) if 1 <= c <= base["max_running_requests"]})
    return [{**base, "chunked_prefill_size": c, "max_running_requests": r} for c in chunks for r in caps]


def build_ladder(
    base: Dict[str, Any],
    wl: Workload,
    slo: SLO,
    hw: Hardware,
    model: ModelSpec,
    params: CalibParams,
    space: SearchSpace,
    measured: Optional[Dict[str, Measurement]] = None,
) -> List[Dict[str, Any]]:
    """Pareto ladder over online knobs, ordered TTFT-priority -> throughput-priority.

    Each rung carries the metrics it was ranked by and ``source``: ``measured`` when
    ``measured`` (config_key -> Measurement) has the exact config, else ``roofline``.
    """
    measured = measured or {}
    rows: List[Dict[str, Any]] = []
    for cfg in online_variants(base, space):
        m = measured.get(config_key(cfg))
        if m is not None and m.get("ok", False):
            row = {"ttft_p90_ms": m["ttft_p90_ms"], "tok_s_slow": m["tok_s_slow"], "throughput_tok_s": m["throughput_tok_s"], "source": "measured"}
        else:
            pr = predict(cfg, wl, hw, model, params)
            if not pr.feasible:
                continue
            row = {"ttft_p90_ms": pr.ttft_p90_ms, "tok_s_slow": pr.tok_s_slow, "throughput_tok_s": pr.total_tok_s, "source": "roofline"}
        row["violation"] = slo.violation(row["ttft_p90_ms"], row["tok_s_slow"])
        row["online"] = {k: cfg[k] for k in ONLINE_KNOBS}
        rows.append(row)
    met = [r for r in rows if r["violation"] <= 0]
    pool = met if met else rows
    ladder = pareto_front(pool, minimise="ttft_p90_ms", maximise="tok_s_slow")
    # ties on the front's metrics: the rung that keeps the larger chunk/cap is more throughput-oriented
    ladder.sort(key=lambda r: (r["ttft_p90_ms"], -r["tok_s_slow"]))
    for i, r in enumerate(ladder):
        r["rung"] = i
        r["slo_met"] = r["violation"] <= 0
    return ladder
