"""The roofline must reproduce the calibration measurements within the file's own bound."""

from pathlib import Path

import pytest

from green_optimizer.roofline import (
    accept_length,
    memory_plan,
    predict,
    prefill_time,
    resolve_calibration,
    ttft_lower_bound,
)
from green_optimizer.spec import ModelSpec, Workload, load_hardware, load_workload

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
model = ModelSpec()


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
