import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prof_breakdown", ROOT / "tools" / "prof_breakdown.py")
pb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb)


def test_classify_decode_with_and_without_spec():
    c = pb.classify
    assert c("fused_recurrent_gated_delta_rule_packed_decode_kernel", "decode", False) == "GDN decode"
    assert c("fused_recurrent_gated_delta_rule_packed_decode_kernel", "decode", True) == "GDN verify"
    assert c("cutlass_flash_attention_forward_sm90_FlashAttentionForwardSm90", "decode", True) == "full-attn verify"
    assert c("FlashAttentionForwardSm90_cutlass", "decode", False) == "full-attn decode"
    assert c("FlashAttentionForwardSm90_cutlass", "extend", True) == "full-attn prefill"
    assert c("chunk_gated_delta_rule_fwd_kkt_solve_kernel", "extend", False) == "GDN extend"
    assert c("causal_conv1d_fwd_kernel", "extend", False) == "GDN extend"
    assert c("nvjet_sm90_tst_128x256", "decode", False) == "GEMM"
    assert c("cutlass3x_sm90_tensorop_gemm", "decode", False) == "GEMM"
    assert c("_fused_qk_rmsnorm_rope_gate_kernel", "decode", False) == "norm/rope/gate/other"


def _trace():
    ev = lambda n, d, cat="kernel", ph="X": {"ph": ph, "cat": cat, "name": n, "dur": d, "ts": 0}
    return {"traceEvents": [
        ev("FlashAttentionForwardSm90", 100), ev("nvjet_x", 300), ev("fused_recurrent_gated_delta_rule_update_fwd_kernel", 50),
        ev("some_cpu_op", 999, cat="cpu_op"), ev("Memcpy HtoD", 10, cat="gpu_memcpy"), ev("rmsnorm_kernel", 40),
        {"ph": "M", "name": "meta"},
    ]}


def test_breakdown_counts_only_kernels_and_memcpy():
    total, names = pb.breakdown(_trace(), "decode", True)
    assert total == {"full-attn verify": 100, "GEMM": 300, "GDN verify": 50, "norm/rope/gate/other": 50}
    assert names["GEMM"] == {"nvjet_x": 300}


def test_main_reads_gz_and_stage_from_name(tmp_path, capsys):
    f = tmp_path / "123-TP-0-EXTEND.trace.json.gz"
    with gzip.open(f, "wt") as g:
        json.dump(_trace(), g)
    out = tmp_path / "o.json"
    assert pb.main([str(tmp_path), "--spec", "--json", str(out)]) == 0
    us = json.loads(out.read_text())["us"]
    assert us["full-attn prefill"] == 100 and us["GDN extend"] == 50
    assert "GEMM" in capsys.readouterr().out
