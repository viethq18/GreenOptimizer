import json
from pathlib import Path

import pytest
import yaml

from green_optimizer.runner import (
    bench_env,
    build_launch,
    load_launch_template,
    parse_profile_export,
    percentile,
    render_launch_script,
    render_scenario,
)
from green_optimizer.spec import SearchSpace, load_workload

ROOT = Path(__file__).resolve().parents[2]
tpl = load_launch_template(str(ROOT / "configs" / "launch.yaml"))
FIXED = {"mem_fraction_static": 0.85}
CFG = {
    "attention_backend": "fa4", "kv_cache_dtype": "fp8_e4m3", "chunked_prefill_size": 8192,
    "max_prefill_tokens": 16384, "max_running_requests": 51, "mamba_ssm_dtype": "float32",
    "mamba_full_memory_ratio": 5, "speculative_num_draft_tokens": 4, "schedule_policy": "dfs-weight", "hicache": True,
}


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_build_launch_maps_knobs_and_spec_and_hicache():
    argv, env = build_launch(CFG, tpl, FIXED)
    assert argv[:3] == [tpl["python"], "-m", "sglang.launch_server"]
    assert _flag(argv, "--attention-backend") == "fa4"
    assert _flag(argv, "--chunked-prefill-size") == "8192"
    assert _flag(argv, "--mem-fraction-static") == "0.85"
    assert _flag(argv, "--speculative-num-draft-tokens") == "4"
    assert _flag(argv, "--speculative-algorithm") == "DFLASH"
    assert "--enable-hierarchical-cache" in argv
    assert "SGLANG_PLUGINS" not in env


def test_build_launch_spec_off_hicache_off_green():
    cfg = {**CFG, "speculative_num_draft_tokens": "off", "hicache": False, "attention_backend": "green"}
    argv, env = build_launch(cfg, tpl, FIXED, port=31000)
    assert "--speculative-algorithm" not in argv and "--enable-hierarchical-cache" not in argv
    assert "--speculative-draft-attention-backend" not in argv  # only with spec
    assert env["SGLANG_PLUGINS"] == "green_attention"
    assert _flag(argv, "--port") == "31000"
    argv2, _ = build_launch({**cfg, "speculative_num_draft_tokens": 4}, tpl, FIXED)
    assert _flag(argv2, "--speculative-draft-attention-backend") == "fa4"


def test_build_launch_green_srpt_maps_to_sglang_policy_and_loads_plugin():
    argv, env = build_launch({**CFG, "schedule_policy": "green-srpt"}, tpl, FIXED)
    assert _flag(argv, "--schedule-policy") == "shortest-prefill-first"
    assert env["GREEN_SRPT"] == "1" and "SGLANG_PLUGINS" not in env  # no whitelist: all plugins load
    _, env = build_launch({**CFG, "schedule_policy": "green-srpt", "attention_backend": "green"}, tpl, FIXED)
    assert env["SGLANG_PLUGINS"] == "green_attention,green_optimizer"
    _, env = build_launch(CFG, tpl, FIXED)
    assert "GREEN_SRPT" not in env


def test_render_launch_script_is_valid_bash(tmp_path):
    import subprocess

    text = render_launch_script(CFG, tpl, FIXED, "test comment")
    f = tmp_path / "l.sh"
    f.write_text(text)
    assert subprocess.run(["bash", "-n", str(f)]).returncode == 0
    assert "--attention-backend fa4" in text and text.startswith("#!/usr/bin/env bash")


def test_render_scenario_request_bound():
    wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
    base = "benchmark:\n  phases:\n  - {name: profiling, duration: 1200}\n  datasets: []\n"
    doc = yaml.safe_load(render_scenario(base, wl, requests=200, duration_s=600))
    ph = doc["benchmark"]["phases"][0]
    assert ph["requests"] == 200 and ph["duration"] == 600.0


def test_bench_env_requires_closed_loop():
    wl = load_workload(str(ROOT / "configs" / "workloads" / "livekit.yaml"))
    env = bench_env(wl, "Qwen/Qwen3.8-27B-FP8", "/tmp/r", "http://127.0.0.1:30000", "/tmp/s.yaml", 30000)
    assert env["CONC"] == str(wl.concurrency) and env["SERVING_SCENARIO_YAML"] == "/tmp/s.yaml"
    assert env["MODEL"] == env["SERVING_TOKENIZER"] == "Qwen/Qwen3.8-27B-FP8"
    open_loop = wl.model_copy(update={"concurrency": None, "arrival_rate": 2.0})
    with pytest.raises(ValueError):
        bench_env(open_loop, "Qwen/Qwen3.8-27B-FP8", "/tmp/r", "http://x", "/tmp/s.yaml", 1)


def test_percentile_matches_numpy_convention():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5.0], 0.9) == 5.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def _rec(phase, start, end, isl, osl, ttft, user, hit=0):
    return {
        "metadata": {"benchmark_phase": phase, "request_start_ns": int(start * 1e9), "request_end_ns": int(end * 1e9)},
        "metrics": {
            "input_sequence_length": {"value": isl}, "output_sequence_length": {"value": osl},
            "time_to_first_token": {"value": ttft}, "output_token_throughput_per_user": {"value": user},
            "usage_prompt_cache_read_tokens": {"value": hit},
        },
    }


def test_parse_profile_export(tmp_path):
    f = tmp_path / "p.jsonl"
    rows = [_rec("warmup", 0, 1, 100, 10, 50, 1)]
    rows += [_rec("profiling", 10 + i, 12 + i, 1000, 10, 100.0 * (i + 1), float(i + 1), 500) for i in range(10)]
    rows.append({"metadata": {"benchmark_phase": "profiling"}, "metrics": {}})  # failed request
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    m = parse_profile_export(str(f))
    assert m["ok"] and m["n_requests"] == 10
    assert m["duration_s"] == pytest.approx(11.0)
    assert m["throughput_tok_s"] == pytest.approx(10 * 1010 / 11.0)
    assert m["ttft_p90_ms"] == pytest.approx(percentile([100.0 * (i + 1) for i in range(10)], 0.9))
    assert m["tok_s_slow"] < m["tok_s_p50"] < m["tok_s_fast"]
    assert m["cache_hit_rate"] == pytest.approx(0.5)
    assert "goodput_req_s" not in m


def test_parse_profile_export_goodput_counts_requests_meeting_both_thresholds(tmp_path):
    from green_optimizer.spec import SLO

    f = tmp_path / "p.jsonl"
    # ttft 100..1000 ms, per-user 1..10 tok/s: only ttft<=500 and tok/s>=5 -> request 5 (ttft 500, 5 tok/s)
    rows = [_rec("profiling", 10 + i, 12 + i, 1000, 10, 100.0 * (i + 1), float(i + 1)) for i in range(10)]
    # a 1-token reply has no per-user speed: judged on TTFT alone
    one = _rec("profiling", 10, 12, 1000, 1, 200.0, 0.0)
    del one["metrics"]["output_token_throughput_per_user"]
    f.write_text("\n".join(json.dumps(r) for r in rows + [one]) + "\n")
    m = parse_profile_export(str(f), SLO(ttft_p90_ms=500, user_tok_s_p90_min=5))
    assert m["slo_attainment"] == pytest.approx(2 / 11)
    assert m["goodput_req_s"] == pytest.approx(2 / m["duration_s"])


def test_parse_profile_export_empty(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text("")
    assert parse_profile_export(str(f))["ok"] is False
