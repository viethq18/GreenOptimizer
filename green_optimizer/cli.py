"""``green-opt`` command line: ``search`` (O1 offline) and ``control`` (O2 online sidecar)."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from . import search as S
from .roofline import Calibration, CalibParams, resolve_calibration, ttft_lower_bound
from .runner import Runner, load_launch_template, render_launch_script
from .spec import (
    ONLINE_KNOBS,
    SLO,
    DEFAULT_SPACE,
    Hardware,
    ModelSpec,
    SearchSpace,
    Workload,
    config_key,
    load_hardware,
    load_slo,
    load_space,
    load_workload,
)

log = logging.getLogger("green_optimizer")

_SHOWN_KNOBS = ("attention_backend", "kv_cache_dtype", "chunked_prefill_size", "max_running_requests", "mamba_ssm_dtype", "mamba_full_memory_ratio", "speculative_num_draft_tokens", "hicache")


def _fmt_cfg(cfg: Dict[str, Any]) -> str:
    return " ".join(f"{k}={cfg[k]}" for k in _SHOWN_KNOBS if k in cfg)


def _json_safe(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _row(c: S.Candidate) -> str:
    p = c.pred
    if not p.feasible:
        return f"| {_fmt_cfg(c.cfg)} | infeasible: {p.reason} | | | | |"
    return (
        f"| {_fmt_cfg(c.cfg)} | {p.ttft_p90_ms:,.0f} | {p.tok_s_slow:.1f} | {p.total_tok_s:,.0f} "
        f"| {p.running:.1f} | {c.violation * 100:.0f}% |"
    )


def write_report(
    out: Path,
    wl: Workload,
    slo: SLO,
    hw: Hardware,
    calib: Calibration,
    space: SearchSpace,
    cands: Sequence[S.Candidate],
    result: Optional[S.SearchResult],
    ladder: Optional[List[Dict[str, Any]]],
    prefilter_only: bool,
    top: int = 15,
) -> str:
    ok = S.passing(cands)
    lb = ttft_lower_bound(wl, hw, ModelSpec(), calib.params, hit=0.0)
    lb_hit = ttft_lower_bound(wl, hw, ModelSpec(), calib.params, hit=wl.hit_rate(True))
    L: List[str] = []
    L.append(f"# GreenOptimizer search report: {wl.name} on {hw.name}\n")
    L.append(f"- SLO: TTFT p90 <= {slo.ttft_p90_ms:g} ms; per-user tok/s >= {slo.user_tok_s_p90_min:g} for 90% of users (slow-tail p10)")
    load = f"closed loop c={wl.concurrency}" if wl.concurrency else f"open loop {wl.arrival_rate} req/s"
    L.append(f"- Workload: {load}, mean ISL {wl.isl.mean():,.0f}, mean OSL {wl.osl.mean():.1f}, prefix hit {wl.prefix_hit_rate:.1%} (hicache) / {wl.hit_rate(False):.1%}")
    L.append(f"- Search space: {space.size():,} raw combinations, {len(cands):,} distinct modelled configs")
    L.append(f"- Calibration: `{hw.calibration}`; roofline accuracy target <= {calib.max_rel_error:.0%} on {', '.join(calib.predicted_metrics)}")
    L.append(f"  - provenance: {calib.provenance.strip()}")
    L.append(f"  - caveat: {calib.caveat.strip()}\n")
    L.append("## Feasibility\n")
    if ok:
        L.append(f"{len(ok)} configs are predicted to meet the SLO within {S.SLACK:.0%} slack.\n")
    else:
        best = cands[0] if cands else None
        L.append("**INFEASIBLE under the roofline model.** No config meets the SLO within the slack.")
        L.append(
            f"- Lower bound of TTFT for a mean request (ISL {lb['isl_mean']:,.0f}): "
            f"{lb['physical_s'] * 1e3:,.0f} ms at the hardware limit with no prefix hit "
            f"({lb['calibrated_s'] * 1e3:,.0f} ms at the calibrated prefill efficiency); "
            f"{lb_hit['physical_s'] * 1e3:,.0f} ms / {lb_hit['calibrated_s'] * 1e3:,.0f} ms at the workload's {wl.prefix_hit_rate:.0%} hit rate. "
            f"The SLO asks {slo.ttft_p90_ms:g} ms for p90, and queueing under load only adds to this unloaded bound."
        )
        if best is not None and best.pred.feasible:
            L.append(
                f"- Closest predicted frontier: `{_fmt_cfg(best.cfg)}` -> TTFT p90 {best.pred.ttft_p90_ms:,.0f} ms, "
                f"slow tok/s {best.pred.tok_s_slow:.1f} (violation {best.violation:.0%})."
            )
        L.append("- Nothing is promised: the numbers above are what the hardware allows; relax the SLO or raise the prefix-hit rate.\n")
    L.append(f"## Roofline ranking (top {top})\n")
    L.append("| config | TTFT p90 ms | slow tok/s | total tok/s | running | SLO violation |")
    L.append("|---|---|---|---|---|---|")
    L += [_row(c) for c in cands[:top]]
    L.append("")
    L.append("## Not modelled by the roofline\n")
    L.append(
        f"`{', '.join(S.UNMODELLED)}` are ranked at the production defaults ({S.UNMODELLED_DEFAULTS}) and decided by measurement only "
        "(a one-at-a-time sweep around the round-2 winner). `mamba_full_memory_ratio` enters only through the memory formula; "
        "`attention_backend=green` uses the A2 acceptance target as its KV-read efficiency (assumption, replace with the measured value once A2 is benchmarked)."
        " The DFLASH acceptance for draft lengths other than the calibrated one comes from a geometric model, not a measurement.\n"
    )
    if prefilter_only:
        L.append("## Measurements\n")
        L.append("Prefilter only (`--prefilter-only`): no server was launched, nothing was measured.\n")
    elif result is not None:
        L.append("## Measurements\n")
        for name, rows in result.rounds.items():
            L.append(f"### {name}\n")
            L.append("| config | ok | TTFT p90 ms | slow tok/s | total tok/s |")
            L.append("|---|---|---|---|---|")
            for cfg, m in rows:
                if m.get("ok"):
                    L.append(f"| {_fmt_cfg(cfg)} | yes | {m['ttft_p90_ms']:,.0f} | {m['tok_s_slow']:.1f} | {m['throughput_tok_s']:,.0f} |")
                else:
                    L.append(f"| {_fmt_cfg(cfg)} | no: {m.get('error', '?')} | | | |")
            L.append("")
        if result.winner is None:
            L.append("**No configuration produced a usable measurement.**\n")
        else:
            wm = result.winner_measurement or {}
            verdict = "meets the SLO" if result.slo_met else "does NOT meet the SLO (best effort)"
            L.append(f"**Winner** ({verdict}): `{_fmt_cfg(result.winner)}`, TTFT p90 {wm.get('ttft_p90_ms', float('nan')):,.0f} ms, "
                     f"slow tok/s {wm.get('tok_s_slow', float('nan')):.1f}, total tok/s {wm.get('throughput_tok_s', float('nan')):,.0f}.\n")
    if ladder:
        L.append("## Online ladder (TTFT priority -> throughput priority)\n")
        L.append("| rung | chunked_prefill_size | max_running cap | TTFT p90 ms | slow tok/s | source | SLO |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ladder:
            L.append(f"| {r['rung']} | {r['online']['chunked_prefill_size']} | {r['online']['max_running_requests']} | {r['ttft_p90_ms']:,.0f} | {r['tok_s_slow']:.1f} | {r['source']} | {'met' if r['slo_met'] else 'violated'} |")
        L.append("")
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    return text


def _choose_base(cands: Sequence[S.Candidate], result: Optional[S.SearchResult]) -> Optional[Dict[str, Any]]:
    if result is not None and result.winner is not None:
        return result.winner
    return cands[0].cfg if cands and cands[0].pred.feasible else None


def cmd_search(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wl, slo, hw = load_workload(a.workload), load_slo(a.slo), load_hardware(a.hw)
    space = load_space(a.space) if a.space else SearchSpace()
    calib = resolve_calibration(hw, a.hw)
    if calib is None:
        print(f"error: {a.hw} has no calibration file; the roofline needs one", file=sys.stderr)
        return 2
    model = ModelSpec()
    cands = S.prefilter(space, wl, slo, hw, model, calib.params)
    ok = S.passing(cands)
    print(f"{space.size():,} raw configs -> {len(cands):,} modelled; {len(ok)} predicted within SLO (+{S.SLACK:.0%})")
    for c in cands[: a.top_k if a.prefilter_only else 10]:
        print(_row(c))
    launch_tpl = load_launch_template(a.launch) if a.launch else None

    result: Optional[S.SearchResult] = None
    measured: Dict[str, Dict[str, Any]] = {}
    if not a.prefilter_only:
        if not ok and a.no_best_effort:
            write_report(out, wl, slo, hw, calib, space, cands, None, None, False)
            print("infeasible under the roofline; see report.md (--no-best-effort: nothing measured)")
            return 1
        if launch_tpl is None:
            print("error: measuring needs --launch <launch template yaml>", file=sys.stderr)
            return 2
        if wl.bench is None:
            print(f"error: workload {wl.name} has no bench block", file=sys.stderr)
            return 2
        runner = Runner(launch_tpl, wl, out, space.fixed)
        pool = ok if ok else cands  # best-effort frontier when nothing passes the prefilter
        results_f = open(out / "results.jsonl", "a")

        def measure(cfg, round_name, repeat):
            reqs = S.ROUND1_REQUESTS if round_name == "round1" else None
            return runner.measure(cfg, round_name, repeat, requests=reqs)

        def on_result(round_name, cfg, m):
            results_f.write(json.dumps(_json_safe({"round": round_name, "config": cfg, "measurement": m})) + "\n")
            results_f.flush()
            if m.get("ok"):
                measured[config_key(cfg)] = m

        try:
            result = S.successive_halving(pool, slo, space, measure, k=a.top_k, on_result=on_result)
        finally:
            results_f.close()

    base = _choose_base(cands, result)
    ladder = None
    if base is not None:
        ladder = S.build_ladder(base, wl, slo, hw, model, calib.params, space, measured)
        with open(out / "ladder.yaml", "w") as f:
            yaml.safe_dump({"base": base, "slo": slo.model_dump(), "ladder": _json_safe(ladder)}, f, sort_keys=False)
        if launch_tpl is not None:
            comment = f"best config for {wl.name} on {hw.name}; generated by green-opt search" + ("" if result and result.slo_met else " (prefilter ranking, unmeasured)" if result is None else " (best effort: SLO not met)")
            (out / "best_launch.sh").write_text(render_launch_script(base, launch_tpl, space.fixed, comment))
    write_report(out, wl, slo, hw, calib, space, cands, result, ladder, a.prefilter_only)
    print(f"wrote {out / 'report.md'}")
    return 0


def cmd_control(a: argparse.Namespace) -> int:
    from .control import Controller, load_ladder

    ladder = load_ladder(a.ladder)
    slo = load_slo(a.slo)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ctl = Controller(a.url, ladder, slo, interval_s=a.interval, window_s=a.window, start_rung=a.start_rung)
    try:
        ctl.run(a.duration)
    except KeyboardInterrupt:
        pass
    if a.events:
        Path(a.events).write_text(json.dumps(_json_safe(ctl.events), indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="green-opt", description="SLO-driven SGLang configuration search and online control")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="offline search: roofline prefilter, successive halving, ladder")
    s.add_argument("--workload", required=True)
    s.add_argument("--slo", required=True)
    s.add_argument("--hw", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--space", help="search-space yaml (default: the plan's knobs)")
    s.add_argument("--launch", help="launch template yaml (configs/launch.yaml); needed to measure and to write best_launch.sh")
    s.add_argument("--top-k", type=int, default=S.ROUND1_K, help="configs entering halving round 1")
    s.add_argument("--prefilter-only", action="store_true", help="rank with the roofline only; launch no server")
    s.add_argument("--no-best-effort", action="store_true", help="abort instead of measuring the closest-to-SLO configs when none is predicted to pass")
    s.set_defaults(fn=cmd_search)
    c = sub.add_parser("control", help="online sidecar: steer along ladder.yaml via /set_internal_state")
    c.add_argument("--url", required=True)
    c.add_argument("--ladder", required=True)
    c.add_argument("--slo", required=True)
    c.add_argument("--interval", type=float, default=5.0)
    c.add_argument("--window", type=float, default=60.0)
    c.add_argument("--start-rung", type=int, default=None)
    c.add_argument("--duration", type=float, default=None, help="stop after this many seconds (default: run until interrupted)")
    c.add_argument("--events", help="write the decision log (json) here on exit")
    c.set_defaults(fn=cmd_control)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
