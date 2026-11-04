"""Offline configuration search: roofline prefilter -> successive halving -> Pareto ladder.

Round structure (plan O1):
  prefilter   roofline over the whole space; drop configs predicted to violate the SLO by
              more than ``slack`` (20%)
  round 1     top-K by the objective (``objective.rank_key``), 200 requests each
  round 2     full bench for the top 25% of round 1
  knob sweep  ``UNMODELLED`` knobs (not in the roofline) varied one at a time around the
              round-2 winner, full bench
  round 3     top 3, repeated 3 times, median of the repeats
The measurement function is injected (``Measure``) so the logic is unit-testable without a GPU.
The default objective ``balanced`` is the original ranking: SLO first, then total tok/s.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .objective import OBJECTIVES, Shape, auto_objective, rank_key
from .roofline import CalibParams, Prediction, predict
from .spec import (
    ONLINE_KNOBS,
    OPT_IN_SPACE,
    SLO,
    Hardware,
    ModelSpec,
    SearchSpace,
    Workload,
    config_key,
)

# knobs the roofline does not model: their effect is decided by measurement only
UNMODELLED = ("max_prefill_tokens", "schedule_policy", *OPT_IN_SPACE)
# default for each unmodelled knob while ranking (the production launch.sh values; sglang's for opt-in knobs)
UNMODELLED_DEFAULTS = {"max_prefill_tokens": 16384, "schedule_policy": "dfs-weight", **{k: v[0] for k, v in OPT_IN_SPACE.items()}}


def unmodelled_defaults(space: SearchSpace) -> Dict[str, Any]:
    """``UNMODELLED_DEFAULTS`` restricted to the knobs ``space`` puts in its configs."""
    return {k: v for k, v in UNMODELLED_DEFAULTS.items() if k not in OPT_IN_SPACE or k in space.space}


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
    rank: Tuple[int, float] = (2, math.inf)  # objective.rank_key of the prediction

    @property
    def key(self) -> str:
        return config_key(self.cfg)

    @property
    def metrics(self) -> Dict[str, Any]:
        p = self.pred
        return {"ok": p.feasible, "ttft_p90_ms": p.ttft_p90_ms, "tok_s_slow": p.tok_s_slow, "throughput_tok_s": p.total_tok_s, "goodput_req_s": p.goodput_req_s}


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
    objective: str = "balanced",
) -> List[Candidate]:
    """Predict every distinct modelled config; sorted best first under ``objective``.

    For ``balanced``: configs within SLO (+slack) first by predicted throughput, then the
    rest by increasing violation (so the head of the list is also the best-effort frontier).
    """
    seen = set()
    out: List[Candidate] = []
    defaults = unmodelled_defaults(space)
    for cfg in space.configs():
        cfg = {**cfg, **defaults}
        if config_key(cfg) in seen:
            continue
        seen.add(config_key(cfg))
        pred = predict(cfg, wl, hw, model, params, slo)
        out.append(Candidate(cfg, pred, _violation(slo, pred)))
    return rerank(out, slo, objective)


def rerank(cands: Sequence[Candidate], slo: SLO, objective: str) -> List[Candidate]:
    for c in cands:
        c.rank = rank_key(objective, slo, c.metrics, SLACK)
    return sorted(cands, key=lambda c: c.rank)


def passing(cands: Sequence[Candidate]) -> List[Candidate]:
    """Configs predicted to meet the SLO within the slack (whatever the objective)."""
    return [c for c in cands if c.pred.feasible and c.violation <= SLACK]


def measure_pool(cands: Sequence[Candidate]) -> List[Candidate]:
    """Configs worth measuring: the objective's acceptable tier, else the best-effort frontier."""
    return [c for c in cands if c.rank[0] == 0] or list(cands)


def choose_objective(mode: str, cands: Sequence[Candidate]) -> Tuple[str, str]:
    """Resolve ``auto`` from the shape of the workload as predicted for the top config of a
    ``balanced`` ranking ``cands``; any other mode is returned unchanged."""
    if mode != "auto":
        return mode, ""
    if not cands or not cands[0].pred.feasible:
        return auto_objective(Shape(0.0, 0.0, 0.0, slo_feasible=False))
    p = cands[0].pred
    return auto_objective(Shape(p.lam * p.wait_ms / 1e3, p.ttft_mean_ms, p.decode_ms, slo_feasible=bool(passing(cands))))


# --------------------------------------------------------------------------- halving
def score_key(slo: SLO, m: Measurement, objective: str = "balanced") -> Tuple[int, float]:
    """Sort key of a measurement (smaller is better), see ``objective.rank_key``."""
    return rank_key(objective, slo, m)


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


def _rank(slo: SLO, rows: List[Tuple[Dict[str, Any], Measurement]], objective: str = "balanced") -> List[Tuple[Dict[str, Any], Measurement]]:
    return sorted(rows, key=lambda r: score_key(slo, r[1], objective))


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
    objective: str = "balanced",
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

    r1 = _rank(slo, run("round1", [c.cfg for c in cands[:k]]), objective)
    n2 = max(top_final, math.ceil(len(r1) * keep_fraction))
    r2 = _rank(slo, run("round2", [cfg for cfg, m in r1[:n2] if m.get("ok", False)]), objective)
    if not r2 or not r2[0][1].get("ok", False):
        return res
    sweep = run("knob_sweep", unmodelled_variants(r2[0][0], space))
    pool = _rank(slo, r2 + sweep, objective)
    finalists = [cfg for cfg, m in pool[:top_final] if m.get("ok", False)]
    final_rows = []
    for cfg in finalists:
        reps = [measure(cfg, "round3", i) for i in range(repeats)]
        for i, m in enumerate(reps):
            if on_result:
                on_result("round3", cfg, {**m, "repeat": i})
        final_rows.append((cfg, median_measurement(reps)))
    res.rounds["round3"] = final_rows
    final_rows = _rank(slo, final_rows, objective)
    if final_rows and final_rows[0][1].get("ok", False):
        res.winner, res.winner_measurement = final_rows[0]
        res.slo_met = slo.violation(res.winner_measurement["ttft_p90_ms"], res.winner_measurement["tok_s_slow"]) <= 0
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
    objective: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pareto ladder over online knobs, ordered TTFT-priority -> throughput-priority.

    Each rung carries the metrics it was ranked by and ``source``: ``measured`` when
    ``measured`` (config_key -> Measurement) has the exact config, else ``roofline``.
    Rungs are drawn from the SLO-meeting variants when there are any, except for the
    ``throughput`` objective, which ignores the SLO. With an ``objective`` the rungs also
    carry ``goodput_req_s`` (for ``ladder_homes``); without one the ladder is the original.
    """
    measured = measured or {}
    rows: List[Dict[str, Any]] = []
    for cfg in online_variants(base, space):
        m = measured.get(config_key(cfg))
        if m is not None and m.get("ok", False):
            row = {"ttft_p90_ms": m["ttft_p90_ms"], "tok_s_slow": m["tok_s_slow"], "throughput_tok_s": m["throughput_tok_s"], "goodput_req_s": m.get("goodput_req_s"), "source": "measured"}
        else:
            pr = predict(cfg, wl, hw, model, params, slo)
            if not pr.feasible:
                continue
            row = {"ttft_p90_ms": pr.ttft_p90_ms, "tok_s_slow": pr.tok_s_slow, "throughput_tok_s": pr.total_tok_s, "goodput_req_s": pr.goodput_req_s, "source": "roofline"}
        row["violation"] = slo.violation(row["ttft_p90_ms"], row["tok_s_slow"])
        row["online"] = {k: cfg[k] for k in ONLINE_KNOBS}
        if objective is None:
            del row["goodput_req_s"]
        rows.append(row)
    met = [r for r in rows if r["violation"] <= 0]
    pool = met if met and objective != "throughput" else rows
    ladder = pareto_front(pool, minimise="ttft_p90_ms", maximise="tok_s_slow")
    # ties on the front's metrics: the rung that keeps the larger chunk/cap is more throughput-oriented
    ladder.sort(key=lambda r: (r["ttft_p90_ms"], -r["tok_s_slow"]))
    for i, r in enumerate(ladder):
        r["rung"] = i
        r["slo_met"] = r["violation"] <= 0
    return ladder


def ladder_homes(ladder: Sequence[Dict[str, Any]], slo: SLO) -> Dict[str, int]:
    """The rung each objective prefers (its ``rank_key`` minimum over the ladder): where the
    controller starts and where it drifts back to while the SLO holds."""
    if not ladder:
        return {}
    return {o: min(ladder, key=lambda r: (rank_key(o, slo, {**r, "ok": True}), r["rung"]))["rung"] for o in OBJECTIVES}
