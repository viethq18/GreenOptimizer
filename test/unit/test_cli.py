from pathlib import Path

import yaml

from green_optimizer import cli

ROOT = Path(__file__).resolve().parents[2]


def _search(out, model, launch):
    return cli.main([
        "search",
        "--workload", str(ROOT / "configs/workloads/livekit.yaml"),
        "--slo", str(ROOT / "configs/slo/example.yaml"),
        "--hw", str(ROOT / "configs/h100.yaml"),
        "--model", str(ROOT / "configs/models" / model),
        "--launch", str(ROOT / "configs" / launch),
        "--out", str(out), "--prefilter-only",
    ])


def test_prefilter_only_writes_report_and_ladder(tmp_path):
    out = tmp_path / "opt"
    rc = _search(out, "qwen3.8-27b-fp8.yaml", "launch.yaml")
    assert rc == 0
    text = (out / "report.md").read_text()
    assert "INFEASIBLE" in text and "Roofline ranking" in text and "Prefilter only" in text
    assert "Not modelled by the roofline" in text
    lad = yaml.safe_load((out / "ladder.yaml").read_text())
    assert lad["ladder"] and "online" in lad["ladder"][0]
    assert (out / "best_launch.sh").exists()
    assert not (out / "results.jsonl").exists()


def test_gemma_search_drops_mamba_and_spec_flags(tmp_path):
    out = tmp_path / "g"
    assert _search(out, "gemma-4-31b-it-fp8-block.yaml", "launch-gemma4.yaml") == 0
    sh = (out / "best_launch.sh").read_text()
    assert "--swa-full-tokens-ratio" in sh
    # only fa4 + bf16 KV and green + fp8 KV can run this model (see the model's `unsupported`)
    assert ("--attention-backend fa4" in sh and "--kv-cache-dtype bfloat16" in sh) or (
        "--attention-backend green" in sh and "--kv-cache-dtype fp8_e4m3" in sh
        and "SGLANG_PLUGINS=green_attention" in sh)
    assert "mamba" not in sh and "--speculative" not in sh
    assert "model mismatch" in (out / "report.md").read_text()


def test_launch_template_of_another_model_is_rejected(tmp_path):
    assert _search(tmp_path / "x", "gemma-4-31b-it-fp8-block.yaml", "launch.yaml") == 2


def test_parser_has_search_and_control():
    p = cli.build_parser()
    a = p.parse_args(["control", "--url", "http://x", "--ladder", "l.yaml", "--slo", "s.yaml"])
    assert a.interval == 5.0 and a.window == 60.0
