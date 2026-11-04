from pathlib import Path

import yaml

from green_optimizer import cli

ROOT = Path(__file__).resolve().parents[2]


def test_prefilter_only_writes_report_and_ladder(tmp_path):
    out = tmp_path / "opt"
    rc = cli.main([
        "search",
        "--workload", str(ROOT / "configs/workloads/livekit.yaml"),
        "--slo", str(ROOT / "configs/slo/example.yaml"),
        "--hw", str(ROOT / "configs/h100.yaml"),
        "--launch", str(ROOT / "configs/launch.yaml"),
        "--out", str(out), "--prefilter-only",
    ])
    assert rc == 0
    text = (out / "report.md").read_text()
    assert "INFEASIBLE" in text and "Roofline ranking" in text and "Prefilter only" in text
    assert "Not modelled by the roofline" in text
    lad = yaml.safe_load((out / "ladder.yaml").read_text())
    assert lad["ladder"] and "online" in lad["ladder"][0]
    assert (out / "best_launch.sh").exists()
    assert not (out / "results.jsonl").exists()


def test_parser_has_search_and_control():
    p = cli.build_parser()
    a = p.parse_args(["control", "--url", "http://x", "--ladder", "l.yaml", "--slo", "s.yaml"])
    assert a.interval == 5.0 and a.window == 60.0
