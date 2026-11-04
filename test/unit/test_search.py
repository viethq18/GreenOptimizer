from pathlib import Path

import pytest

from green_optimizer import search as S
from green_optimizer.roofline import resolve_calibration
from green_optimizer.spec import (
    SLO,
    load_model,
    SearchSpace,
    config_key,
    load_hardware,
    load_slo,
    load_workload,
)

ROOT = Path(__file__).resolve().parents[2]
hw = load_hardware(str(ROOT / "configs" / "h100.yaml"))
calib = resolve_calibration(hw, str(ROOT / "configs" / "h100.yaml"))
wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
slo = load_slo(str(ROOT / "configs" / "slo" / "example.yaml"))
SLO_ = SLO(ttft_p90_ms=1000, user_tok_s_p90_min=50)
QWEN = load_model(str(ROOT / "configs" / "models" / "qwen3.8-27b-fp8.yaml"))


def test_pareto_front_drops_dominated_and_duplicates():
    pts = [
        {"t": 100, "x": 10}, {"t": 200, "x": 30}, {"t": 150, "x": 5},
        {"t": 200, "x": 30}, {"t": 300, "x": 25}, {"t": 400, "x": 40},
    ]
    f = S.pareto_front(pts, "t", "x")
    assert [(p["t"], p["x"]) for p in f] == [(100, 10), (200, 30), (400, 40)]


def test_score_key_orders_met_by_throughput_then_by_violation():
    a = {"ok": True, "ttft_p90_ms": 800, "tok_s_slow": 60, "throughput_tok_s": 100}
    b = {"ok": True, "ttft_p90_ms": 900, "tok_s_slow": 60, "throughput_tok_s": 200}
    c = {"ok": True, "ttft_p90_ms": 2000, "tok_s_slow": 60, "throughput_tok_s": 999}
    d = {"ok": True, "ttft_p90_ms": 1500, "tok_s_slow": 60, "throughput_tok_s": 999}
    bad = {"ok": False}
    order = sorted([a, b, c, d, bad], key=lambda m: S.score_key(SLO_, m))
    assert order == [b, a, d, c, bad]


def test_objectives_pick_different_winners():
    a = {"ok": True, "ttft_p90_ms": 800, "tok_s_slow": 60, "throughput_tok_s": 100, "goodput_req_s": 5.0}
    b = {"ok": True, "ttft_p90_ms": 900, "tok_s_slow": 80, "throughput_tok_s": 200, "goodput_req_s": 3.0}
    c = {"ok": True, "ttft_p90_ms": 300, "tok_s_slow": 55, "throughput_tok_s": 50, "goodput_req_s": 4.0}
    v = {"ok": True, "ttft_p90_ms": 2000, "tok_s_slow": 60, "throughput_tok_s": 999, "goodput_req_s": 6.0}
    best = {o: min([a, b, c, v], key=lambda m: S.score_key(SLO_, m, o)) for o in ("balanced", "ttft", "itl", "goodput", "throughput")}
    assert best == {"balanced": b, "ttft": c, "itl": b, "goodput": v, "throughput": v}
    # SLO-tiered modes never prefer a violating config; goodput without a value falls back to the violation
    assert S.score_key(SLO_, v, "ttft")[0] == 1
    assert S.score_key(SLO_, {**v, "goodput_req_s": None}, "goodput") == (1, SLO_.violation(2000, 60))


def test_ladder_homes_follow_each_objective():
    rows = [
        {"rung": 0, "ttft_p90_ms": 300, "tok_s_slow": 55, "throughput_tok_s": 50, "goodput_req_s": 2.0},
        {"rung": 1, "ttft_p90_ms": 600, "tok_s_slow": 70, "throughput_tok_s": 300, "goodput_req_s": 4.0},
        {"rung": 2, "ttft_p90_ms": 900, "tok_s_slow": 90, "throughput_tok_s": 250, "goodput_req_s": 3.0},
    ]
    assert S.ladder_homes(rows, SLO_) == {"balanced": 1, "ttft": 0, "itl": 2, "goodput": 1, "throughput": 1}


def test_median_measurement_ignores_failures():
    ms = [{"ok": True, "x": 1.0}, {"ok": False, "error": "boom"}, {"ok": True, "x": 3.0}, {"ok": True, "x": 2.0}]
    m = S.median_measurement(ms)
    assert m["ok"] and m["x"] == 2.0 and m["repeats"] == 3
    assert not S.median_measurement([{"ok": False, "error": "e"}])["ok"]


def _small_space():
    return SearchSpace(
        space={
            "attention_backend": ["fa4", "green"],
            "kv_cache_dtype": ["fp8_e4m3"],
            "chunked_prefill_size": [4096, 8192],
            "max_prefill_tokens": [8192, 16384],
            "max_running_requests": [32, 64],
            "mamba_ssm_dtype": ["float32"],
            "mamba_full_memory_ratio": [5],
            "speculative_num_draft_tokens": ["off", 4],
            "schedule_policy": ["dfs-weight", "fcfs"],
            "hicache": [True],
        },
        fixed={"mem_fraction_static": 0.85},
    ).for_model(QWEN)


def test_prefilter_dedupes_unmodelled_knobs_and_sorts():
    sp = _small_space()
    cands = S.prefilter(sp, wl, slo, hw, QWEN, calib.params)
    assert len(cands) == sp.size() // 4  # max_prefill_tokens x schedule_policy collapse
    keys = {config_key(c.cfg) for c in cands}
    assert len(keys) == len(cands)
    ranks = [c.rank for c in cands]
    assert ranks == sorted(ranks)


def test_prefilter_example_slo_is_infeasible_on_livekit():
    cands = S.prefilter(_small_space(), wl, slo, hw, QWEN, calib.params)
    assert S.passing(cands) == []  # TTFT p90 < 500 ms at ISL 21k is not reachable


def test_successive_halving_with_fake_measure():
    sp = _small_space()
    cands = S.prefilter(sp, wl, slo, hw, QWEN, calib.params)
    calls = []

    def measure(cfg, round_name, repeat):
        calls.append((round_name, config_key(cfg), repeat))
        if cfg["attention_backend"] == "fa4" and cfg["chunked_prefill_size"] == 4096:
            return {"ok": False, "error": "oom"}
        score = 100 + (50 if cfg["attention_backend"] == "green" else 0) + cfg["max_running_requests"]
        if cfg["schedule_policy"] == "fcfs":
            score -= 10
        return {"ok": True, "ttft_p90_ms": 400.0, "tok_s_slow": 120.0, "tok_s_fast": 150.0, "throughput_tok_s": float(score)}

    seen = []
    res = S.successive_halving(cands, SLO_, sp, measure, k=8, on_result=lambda r, c, m: seen.append(r))
    assert set(res.rounds) == {"round1", "round2", "knob_sweep", "round3"}
    assert len(res.rounds["round1"]) == 8
    assert len(res.rounds["round2"]) == 3 or len(res.rounds["round2"]) == 2
    assert sum(1 for c in calls if c[0] == "round3") == 3 * len(res.rounds["round3"])
    assert res.winner is not None and res.slo_met
    assert res.winner["attention_backend"] == "green" and res.winner["max_running_requests"] == 64
    assert res.winner["schedule_policy"] == "dfs-weight"  # the sweep found fcfs worse
    assert seen.count("round3") == 3 * len(res.rounds["round3"])


def test_successive_halving_all_failures_has_no_winner():
    sp = _small_space()
    cands = S.prefilter(sp, wl, slo, hw, QWEN, calib.params)
    res = S.successive_halving(cands, SLO_, sp, lambda c, r, i: {"ok": False, "error": "x"}, k=4)
    assert res.winner is None and not res.slo_met


def test_ladder_is_sorted_and_varies_only_online_knobs():
    sp = _small_space()
    base = S.prefilter(sp, wl, slo, hw, QWEN, calib.params)[0].cfg
    ladder = S.build_ladder(base, wl, slo, hw, QWEN, calib.params, sp, {})
    assert ladder
    ttfts = [r["ttft_p90_ms"] for r in ladder]
    thr = [r["tok_s_slow"] for r in ladder]  # per-user speed: the throughput-priority axis
    assert ttfts == sorted(ttfts) and thr == sorted(thr)
    assert [r["rung"] for r in ladder] == list(range(len(ladder)))
    for r in ladder:
        assert set(r["online"]) == {"chunked_prefill_size", "max_running_requests"}
        assert r["online"]["chunked_prefill_size"] <= base["chunked_prefill_size"]
        assert r["online"]["max_running_requests"] <= base["max_running_requests"]
