import pytest

from green_optimizer.control import (
    ApplyError,
    Controller,
    LadderPolicy,
    Rung,
    apply_rung,
    bucket_delta,
    histogram_count,
    histogram_quantile,
    load_ladder,
    parse_histogram,
    server_args_for,
)
from green_optimizer.spec import SLO

SLO_ = SLO(ttft_p90_ms=500, user_tok_s_p90_min=100)

TEXT = """\
# HELP sglang:time_to_first_token_seconds ttft
# TYPE sglang:time_to_first_token_seconds histogram
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="true",le="0.1"} 10.0
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="true",le="0.5"} 90.0
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="true",le="1.0"} 100.0
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="true",le="+Inf"} 100.0
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="false",le="0.1"} 5.0
sglang:time_to_first_token_seconds_bucket{model_name="m",is_streaming="false",le="+Inf"} 5.0
sglang:time_to_first_token_seconds_count{model_name="m",is_streaming="true"} 100.0
"""


def test_parse_histogram_filters_labels_and_sums_others():
    b = parse_histogram(TEXT, "sglang:time_to_first_token_seconds", {"is_streaming": "true"})
    assert b == {0.1: 10.0, 0.5: 90.0, 1.0: 100.0, float("inf"): 100.0}
    allb = parse_histogram(TEXT, "sglang:time_to_first_token_seconds")
    assert allb[0.1] == 15.0 and histogram_count(allb) == 105.0


def test_p90_interpolates_inside_bucket():
    b = parse_histogram(TEXT, "sglang:time_to_first_token_seconds", {"is_streaming": "true"})
    assert histogram_quantile(b, 0.9) == pytest.approx(0.5)  # rank 90 = upper edge of le=0.5
    assert histogram_quantile(b, 0.5) == pytest.approx(0.1 + 0.4 * 40 / 80)
    assert histogram_quantile({}, 0.9) is None
    assert histogram_quantile({0.1: 0.0, float("inf"): 0.0}, 0.9) is None


def test_bucket_delta_handles_counter_reset():
    new, old = {1.0: 5.0, float("inf"): 5.0}, {1.0: 3.0, float("inf"): 3.0}
    assert bucket_delta(new, old) == {1.0: 2.0, float("inf"): 2.0}
    assert bucket_delta(old, new) == old  # decreased -> server restarted


def test_policy_needs_two_consecutive_windows():
    p = LadderPolicy(3, start=1, slo=SLO_)
    assert p.step(900, 150).action == "hold"
    d = p.step(900, 150)
    assert (d.action, d.rung) == ("to_ttft", 0)
    # a good window in between resets the streak
    p = LadderPolicy(3, start=1, slo=SLO_)
    p.step(900, 150)
    p.step(100, 150)
    assert p.step(900, 150).action == "hold"


def test_policy_throughput_direction_and_bounds():
    p = LadderPolicy(2, start=0, slo=SLO_)
    p.step(100, 50)
    d = p.step(100, 50)
    assert (d.action, d.rung) == ("to_throughput", 1)
    p.step(100, 50)
    assert p.step(100, 50).action == "hold"  # already at the throughput end


def test_policy_both_violated_holds_and_reports_infeasible():
    p = LadderPolicy(3, start=1, slo=SLO_)
    for _ in range(4):
        d = p.step(900, 50)
        assert (d.action, d.rung) == ("infeasible", 1)


def test_policy_ignores_missing_metrics():
    p = LadderPolicy(3, start=1, slo=SLO_)
    for _ in range(5):
        assert p.step(None, None).action == "hold"


def test_apply_rung_raises_when_not_updated():
    r = Rung(0, {"chunked_prefill_size": 4096, "max_running_requests": 32})
    sent = {}

    def post(url, payload):
        sent["url"], sent["payload"] = url, payload
        return {"updated": True}

    apply_rung("http://x:1/", r, post)
    assert sent["url"] == "http://x:1/set_internal_state"
    assert sent["payload"] == {"server_args": {"green_chunked_prefill_size": 4096, "green_max_running_cap": 32}}
    with pytest.raises(ApplyError):
        apply_rung("http://x:1", r, lambda u, p: {"updated": False})
    assert server_args_for(r)["green_max_running_cap"] == 32


def _metrics(t_count, ttft_slow, itl_slow):
    """Cumulative text: t_count requests, ttft_slow of them in the 2 s bucket, same for ITL."""
    fast = t_count - ttft_slow
    ifast = t_count - itl_slow
    return (
        f'sglang:time_to_first_token_seconds_bucket{{is_streaming="true",le="0.5"}} {fast}\n'
        f'sglang:time_to_first_token_seconds_bucket{{is_streaming="true",le="2.0"}} {t_count}\n'
        f'sglang:time_to_first_token_seconds_bucket{{is_streaming="true",le="+Inf"}} {t_count}\n'
        f'sglang:inter_token_latency_seconds_bucket{{le="0.005"}} {ifast}\n'
        f'sglang:inter_token_latency_seconds_bucket{{le="0.05"}} {t_count}\n'
        f'sglang:inter_token_latency_seconds_bucket{{le="+Inf"}} {t_count}\n'
    )


def test_controller_moves_after_two_violating_windows_and_logs():
    ladder = [
        Rung(0, {"chunked_prefill_size": 2048, "max_running_requests": 16}),
        Rung(1, {"chunked_prefill_size": 8192, "max_running_requests": 51}),
    ]
    now = [0.0]
    posts = []
    texts = iter([_metrics(100, 0, 0), _metrics(200, 50, 0), _metrics(300, 100, 0), _metrics(400, 150, 0)])
    ctl = Controller(
        "http://h:1", ladder, SLO_, interval_s=5, window_s=60,
        get=lambda url: next(texts), post=lambda u, p: posts.append(p) or {"updated": True},
        clock=lambda: now[0], sleep=lambda s: None,
    )
    ctl.start()
    assert posts[0]["server_args"]["green_chunked_prefill_size"] == 8192  # starts at throughput end
    actions = []
    for _ in range(4):
        now[0] += 5
        actions.append(ctl.tick().action)
    assert actions[:3] == ["hold", "hold", "to_ttft"] or "to_ttft" in actions
    assert posts[-1]["server_args"]["green_chunked_prefill_size"] == 2048
    assert any(e["event"] == "apply" and e["why"] == "to_ttft" for e in ctl.events)


def test_load_ladder_yaml(tmp_path):
    f = tmp_path / "l.yaml"
    f.write_text("ladder:\n- {rung: 0, online: {chunked_prefill_size: 2048, max_running_requests: 8}, ttft_p90_ms: 400, tok_s_slow: 90}\n- {online: {chunked_prefill_size: 8192, max_running_requests: 51}}\n")
    r = load_ladder(str(f))
    assert [x.rung for x in r] == [0, 1] and r[0].ttft_p90_ms == 400
