"""The roofline must reproduce the calibration measurements within the file's own bound."""

from pathlib import Path

import pytest

from green_optimizer.roofline import (
    accept_length,
    attended_keys,
    memory_plan,
    predict,
    prefill_time,
    resolve_calibration,
    ttft_lower_bound,
)
from green_optimizer.spec import SLO, Workload, load_hardware, load_model, load_workload

ROOT = Path(__file__).resolve().parents[2]
HW_PATH = str(ROOT / "configs" / "h100.yaml")
FIELD = {
    "req_s": "lam",
    "ttft_mean_ms": "ttft_mean_ms",
    "ttft_p90_ms": "ttft_p90_ms",
    "running": "running",
    "tok_s_fast": "tok_s_fast",
    "tok_s_slow": "tok_s_slow",
    "max_running_effective": "max_running_effective",
    "kv_tokens": "kv_tokens",
}

hw = load_hardware(HW_PATH)
calib = resolve_calibration(hw, HW_PATH)
model = load_model(str(ROOT / "configs" / "models" / "qwen3.8-27b-fp8.yaml"))
gemma = load_model(str(ROOT / "configs" / "models" / "gemma-4-31b-it-fp8-block.yaml"))
GEMMA_CFG = {
    "attention_backend": "fa4", "kv_cache_dtype": "bfloat16", "chunked_prefill_size": 8192,
    "max_prefill_tokens": 16384, "max_running_requests": 64, "swa_full_tokens_ratio": 0.8,
    "schedule_policy": "dfs-weight", "hicache": True, "mem_fraction_static": 0.85,
}


def _cases():
    return [(r.name, k, v) for r in calib.runs for k, v in r.observed.items() if k in FIELD]


def test_calibration_present_and_covers_predicted_metrics():
    assert calib is not None and calib.max_rel_error <= 0.20
    observed = {k for r in calib.runs for k in r.observed}
    assert set(calib.predicted_metrics) <= observed
    assert {r.role for r in calib.runs} == {"fitted", "held_out"}


@pytest.mark.parametrize("run_name,key,obs", _cases())
def test_prediction_within_calibration_error(run_name, key, obs):
    run = next(r for r in calib.runs if r.name == run_name)
    pred = predict(run.config, Workload(**run.workload), hw, model, calib.params)
    assert pred.feasible, pred.reason
    val = getattr(pred, FIELD[key])
    assert abs(val - obs) / obs <= calib.max_rel_error, f"{key}: predicted {val} observed {obs}"


def test_memory_plan_matches_production_server():
    run = calib.runs[0]
    m = memory_plan(run.config, hw, model, calib.params)
    assert m["kv_tokens"] == pytest.approx(117725, rel=0.05)
    assert m["max_requests_by_mamba"] == pytest.approx(51, abs=2)


def test_bf16_state_frees_memory_for_more_slots():
    run = calib.runs[0]
    bf16 = {**run.config, "mamba_ssm_dtype": "bfloat16"}
    assert memory_plan(bf16, hw, model, calib.params)["max_requests_by_mamba"] > memory_plan(run.config, hw, model, calib.params)["max_requests_by_mamba"]


def test_accept_length_monotone_in_draft_tokens():
    cfg = dict(calib.runs[0].config)
    vals = [accept_length({**cfg, "speculative_num_draft_tokens": d}, calib.params) for d in ("off", 2, 4, 8)]
    assert vals[0] == 1.0 and vals == sorted(vals)


def test_slo_attainment_bounded_and_monotone_in_slo():
    run = calib.runs[0]
    wl = Workload(**run.workload)
    no_slo = predict(run.config, wl, hw, model, calib.params)
    assert no_slo.slo_attainment is None and no_slo.goodput_req_s is None
    att = []
    for scale in (0.25, 0.5, 1.0, 2.0, 8.0):
        slo = SLO(ttft_p90_ms=no_slo.ttft_p90_ms * scale, user_tok_s_p90_min=no_slo.tok_s_slow / scale)
        p = predict(run.config, wl, hw, model, calib.params, slo)
        assert 0.0 <= p.slo_attainment <= 1.0
        assert p.goodput_req_s == pytest.approx(p.slo_attainment * p.lam)
        att.append(p.slo_attainment)
    assert att == sorted(att) and att[0] < att[-1]


def test_prefill_time_grows_with_prefix_and_tokens():
    a = prefill_time(8192, 0, hw, model, 0.47)
    assert prefill_time(8192, 16000, hw, model, 0.47) > a
    assert prefill_time(16384, 0, hw, model, 0.47) > a


def test_ttft_lower_bound_livekit_without_hit():
    wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
    lb = ttft_lower_bound(wl, hw, model, calib.params, hit=0.0)
    assert 0.4 < lb["physical_s"] < 1.0  # hardware limit (eta=1, peak FP8)
    assert 1.0 < lb["calibrated_s"] < 2.5  # at the measured prefill efficiency; far above the 500 ms SLO
    assert lb["calibrated_s"] > 0.5 * 2
    assert ttft_lower_bound(wl, hw, model, calib.params, hit=0.5)["physical_s"] < lb["physical_s"]


def test_gemma_memory_plan_reproduces_sglang_hybrid_swa_pool():
    # runs/gemma4/fa4_try1.log: fp8 KV, KV budget 30.04 GiB -> 87485 full / 69988 SWA tokens
    p = calib.params.model_copy(update={"reserved_gib": 0.85 * hw.hbm_gib - gemma.weight_gib_resident - 30.04})
    m = memory_plan({**GEMMA_CFG, "kv_cache_dtype": "fp8_e4m3"}, hw, gemma, p)
    assert m["kv_tokens"] == pytest.approx(87485, rel=1e-3)
    assert m["swa_tokens"] == int(m["kv_tokens"] * 0.8)
    assert m["mamba_slots"] == 0 and m["max_requests_by_mamba"] is None


def test_attended_keys_window_boundaries():
    assert attended_keys(100, 0, None) == attended_keys(100, 0, 10**6) == 5000
    assert attended_keys(100, 2000, 1024) == 100 * 1024  # whole chunk past the window
    # additive across the window edge: [900, 1100) == [900, 1024) + [1024, 1100)
    assert attended_keys(200, 900, 1024) == pytest.approx(attended_keys(124, 900, 1024) + attended_keys(76, 1024, 1024))
    assert attended_keys(200, 900, 1024) < attended_keys(200, 900, None)


def test_gemma_runnable_backend_kv_pairs():
    wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
    fa4_fp8 = predict({**GEMMA_CFG, "kv_cache_dtype": "fp8_e4m3"}, wl, hw, gemma, calib.params)
    assert not fa4_fp8.feasible and "head_dim <= 256" in fa4_fp8.reason
    green_bf16 = predict({**GEMMA_CFG, "attention_backend": "green"}, wl, hw, gemma, calib.params)
    assert not green_bf16.feasible and "fp8-KV only" in green_bf16.reason
    green_fp8 = {**GEMMA_CFG, "attention_backend": "green", "kv_cache_dtype": "fp8_e4m3"}
    assert predict(green_fp8, wl, hw, gemma, calib.params).feasible
    assert predict(GEMMA_CFG, wl, hw, gemma, calib.params).feasible


def test_gemma_swa_pool_below_window_is_infeasible_and_small_pool_caps_running():
    wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
    tiny = predict({**GEMMA_CFG, "swa_full_tokens_ratio": 0.001}, wl, hw, gemma, calib.params)
    assert not tiny.feasible and "SWA pool" in tiny.reason
    small = predict({**GEMMA_CFG, "swa_full_tokens_ratio": 0.01, "max_running_requests": 128}, wl, hw, gemma, calib.params)
    swa_per_req = 1024 + 128  # min(ctx, window) + eviction interval; livekit ctx >> window
    assert small.feasible and small.running <= 0.6 * small.swa_tokens / swa_per_req
    balanced = predict({**GEMMA_CFG, "swa_full_tokens_ratio": 0.05, "max_running_requests": 128}, wl, hw, gemma, calib.params)
    assert balanced.running > small.running
