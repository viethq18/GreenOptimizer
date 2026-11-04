"""Launch an SGLang server for a config, drive the InferenceX/aiperf client, parse the result.

Everything that does not touch a process (command building, scenario rendering, result
parsing) is a pure function so it is unit-tested on CPU; ``Runner`` glues them to real
processes and needs a free GPU.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .plugin import SRPT_BASE_POLICY, SRPT_ENV, SRPT_POLICY
from .spec import SLO, SPEC_OFF, Workload

log = logging.getLogger("green_optimizer.runner")


# --------------------------------------------------------------------------- launch command
def _flag_args(flags: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for k, v in flags.items():
        if v is True:
            out.append(k)
        elif v is False or v is None:
            continue
        else:
            out += [k, str(v)]
    return out


def load_launch_template(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_launch(cfg: Dict[str, Any], tpl: Dict[str, Any], fixed: Optional[Dict[str, Any]] = None, port: Optional[int] = None) -> Tuple[List[str], Dict[str, str]]:
    """(argv, env overrides) for ``python -m sglang.launch_server`` of ``cfg``."""
    port = port or int(tpl["port"])
    values = {**(fixed or {}), **cfg}
    flags: Dict[str, Any] = dict(tpl["args"])
    for knob, flag in tpl["knob_flags"].items():
        if knob in values:
            flags[flag] = values[knob]
    env = {k: str(v) for k, v in tpl.get("env", {}).items()}
    spec_on = values.get("speculative_num_draft_tokens", SPEC_OFF) != SPEC_OFF
    if spec_on:
        flags.update(tpl["spec_args"])
        flags[tpl["spec_draft_flag"]] = values["speculative_num_draft_tokens"]
    if values.get("hicache") in ("on", True):
        flags.update(tpl["hicache_args"])
    if values.get("attention_backend") == "green":
        if spec_on:
            flags.update(tpl.get("green_args", {}))
        env.update({k: str(v) for k, v in tpl.get("green_env", {}).items()})
    if values.get("schedule_policy") == SRPT_POLICY:
        # green-srpt = SGLang's shortest-prefill-first + this package's plugin hooks
        flags[tpl["knob_flags"]["schedule_policy"]] = SRPT_BASE_POLICY
        env[SRPT_ENV] = "1"
        if env.get("SGLANG_PLUGINS"):  # a whitelist must let the plugin load
            env["SGLANG_PLUGINS"] += ",green_optimizer"
    if values.get("prefill_delayer_queue_min_ratio") is not None:
        flags["--enable-prefill-delayer"] = True  # the queue-ratio trigger is read only with the delayer on
    flags["--port"] = port
    argv = [tpl["python"], "-m", "sglang.launch_server", *_flag_args(flags)]
    return argv, env


def render_launch_script(cfg: Dict[str, Any], tpl: Dict[str, Any], fixed: Optional[Dict[str, Any]] = None, comment: str = "") -> str:
    """Standalone bash script (best_launch.sh)."""
    argv, env = build_launch(cfg, tpl, fixed)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    if comment:
        lines += [f"# {c}" for c in comment.splitlines()]
    for k, v in {**tpl.get("env", {}), **env}.items():
        lines.append(f"export {k}={shlex.quote(str(v))}")
    lines.append(f"cd {shlex.quote(str(tpl.get('cwd', '/')))}")
    flags = argv[3:]
    rendered = " ".join(shlex.quote(a) for a in flags)
    lines.append(f"exec {shlex.quote(argv[0])} -m sglang.launch_server {rendered} \"$@\"")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- scenario
def render_scenario(base_yaml: str, wl: Workload, requests: Optional[int] = None, duration_s: Optional[float] = None) -> str:
    """Scenario YAML for one measurement: ``requests`` makes the profiling phase request-
    bounded (halving round 1); ``duration_s`` caps it."""
    doc = yaml.safe_load(base_yaml)
    bench = wl.bench
    assert bench is not None
    phase = doc["benchmark"]["phases"][0]
    if requests is not None:
        phase["requests"] = int(requests)
    if duration_s is not None:
        phase["duration"] = float(duration_s)
    if bench.synthetic_isl is not None:
        for ds in doc["benchmark"]["datasets"]:
            if ds.get("type") == "synthetic":
                ds["isl"] = bench.synthetic_isl
                if bench.synthetic_osl is not None:
                    ds["osl"] = bench.synthetic_osl
    return yaml.safe_dump(doc, sort_keys=False)


def served_model_name(tpl: Dict[str, Any]) -> str:
    return str(tpl["args"]["--served-model-name"])


def bench_env(wl: Workload, model: str, result_dir: str, url: str, scenario: str, port: int) -> Dict[str, str]:
    """``model`` = the server's --served-model-name (also used as the client tokenizer)."""
    b = wl.bench
    assert b is not None
    if wl.concurrency is None:
        raise ValueError("the InferenceX client is driven closed-loop (concurrency); open-loop workloads cannot be measured with it")
    return {
        "PATH": "/usr/local/bin:" + os.environ.get("PATH", ""),
        "PYTHONPATH": b.infx_dir,
        "AIPERF_RUNTIME_DIR": b.aiperf_runtime_dir,
        "RESULT_DIR": result_dir,
        "AGENTIC_OUTPUT_DIR": result_dir,
        "RESULT_FILENAME": "result",
        "KV_OFFLOADING": "none",
        "MODEL": model,
        "PORT": str(port),
        "CONC": str(wl.concurrency),
        "SERVING_REMOTE_URL": url,
        "SERVING_TOKENIZER": model,
        "SERVING_DATASET_KIND": b.dataset_kind,
        "SERVING_DATASET": b.dataset,
        "SERVING_SCENARIO_LABEL": b.scenario_label,
        "SERVING_SEARCH": "0",
        "SERVING_SCENARIO_YAML": scenario,
        "AIPERF_SERVER_METRICS_URLS": f"{url}/metrics",
    }


# --------------------------------------------------------------------------- result parsing
def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolated q-quantile (q in [0,1]) like numpy's default."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty sample")
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _metric(rec: Dict[str, Any], name: str) -> Optional[float]:
    m = rec.get("metrics", {}).get(name)
    if m is None:
        return None
    v = m.get("value") if isinstance(m, dict) else m
    return float(v) if isinstance(v, (int, float)) else None


def parse_profile_export(path: str, slo: Optional[SLO] = None) -> Dict[str, Any]:
    """Summarise an aiperf ``profile_export.jsonl`` (profiling-phase records only).

    ``tok_s_slow`` is the 10th percentile of per-user decode speed (SLO semantics: 90% of
    users get at least this) and ``tok_s_fast`` its 90th percentile (aiperf's own "p90").
    With ``slo``: ``slo_attainment`` is the fraction of requests whose own TTFT and decode
    speed both meet the thresholds (a request without a decode speed, e.g. one output
    token, is judged on TTFT alone) and ``goodput_req_s`` those requests per second."""
    ttft, per_user, tokens = [], [], 0.0
    n = good = 0
    t0, t1 = float("inf"), 0.0
    hit_tokens = in_tokens = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("metadata", {}).get("benchmark_phase", "profiling") != "profiling":
                continue
            isl = _metric(rec, "input_sequence_length")
            osl = _metric(rec, "output_sequence_length")
            tt = _metric(rec, "time_to_first_token")
            if tt is None or isl is None or osl is None:
                continue  # failed / cancelled request
            n += 1
            ttft.append(tt)
            tokens += isl + osl
            in_tokens += isl
            hit_tokens += _metric(rec, "usage_prompt_cache_read_tokens") or 0.0
            u = _metric(rec, "output_token_throughput_per_user")
            if u is not None:
                per_user.append(u)
            if slo is not None and tt <= slo.ttft_p90_ms and (u is None or u >= slo.user_tok_s_p90_min):
                good += 1
            md = rec["metadata"]
            if "request_start_ns" in md:
                t0 = min(t0, md["request_start_ns"] / 1e9)
            if "request_end_ns" in md:
                t1 = max(t1, md["request_end_ns"] / 1e9)
    if n == 0 or not per_user or t1 <= t0:
        return {"ok": False, "error": f"no usable profiling records in {path} (n={n})"}
    dur = t1 - t0
    out = {
        "ok": True,
        "n_requests": n,
        "duration_s": dur,
        "req_s": n / dur,
        "throughput_tok_s": tokens / dur,
        "ttft_p50_ms": percentile(ttft, 0.5),
        "ttft_p90_ms": percentile(ttft, 0.9),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tok_s_slow": percentile(per_user, 0.10),
        "tok_s_p50": percentile(per_user, 0.50),
        "tok_s_fast": percentile(per_user, 0.90),
        "cache_hit_rate": hit_tokens / in_tokens if in_tokens else 0.0,
    }
    if slo is not None:
        out["slo_attainment"] = good / n
        out["goodput_req_s"] = good / dur
    return out


# --------------------------------------------------------------------------- processes
def kill_server(proc: Optional[subprocess.Popen], port: int) -> None:
    """Stop the server this runner started. The process group is signalled first; the
    pkill is scoped to this port so another user's server on the shared host survives."""
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
    subprocess.run(["pkill", "-f", f"[s]glang.launch_server.*--port {port}"], check=False)
    time.sleep(5)


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class Runner:
    def __init__(self, tpl: Dict[str, Any], wl: Workload, out_dir: Path, fixed: Optional[Dict[str, Any]] = None, slo: Optional[SLO] = None):
        self.tpl = tpl
        self.wl = wl
        self.out_dir = Path(out_dir)
        self.fixed = fixed or {}
        self.slo = slo
        self.port = int(tpl["port"])
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None
        self._counter = 0

    def start_server(self, cfg: Dict[str, Any], log_path: Path) -> None:
        argv, env = build_launch(cfg, self.tpl, self.fixed, self.port)
        kill_server(None, self.port)
        logf = open(log_path, "w")
        self.proc = subprocess.Popen(
            argv,
            cwd=self.tpl.get("cwd", "/"),
            env={**os.environ, **{k: str(v) for k, v in self.tpl.get("env", {}).items()}, **env},
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + float(self.tpl.get("ready_timeout_s", 1800))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with {self.proc.returncode}; see {log_path}")
            if http_ok(f"{self.url}/health"):
                return
            time.sleep(5)
        raise TimeoutError(f"server not ready after {self.tpl.get('ready_timeout_s')} s; see {log_path}")

    def stop_server(self) -> None:
        kill_server(self.proc, self.port)
        self.proc = None

    def bench(self, requests: Optional[int], tag: str) -> Dict[str, Any]:
        b = self.wl.bench
        assert b is not None
        result_dir = self.out_dir / "bench" / tag
        result_dir.mkdir(parents=True, exist_ok=True)
        base = (Path(b.infx_dir) / b.scenario_yaml).read_text()
        scenario = render_scenario(base, self.wl, requests=requests, duration_s=b.max_duration_s)
        env = {**os.environ, **bench_env(self.wl, served_model_name(self.tpl), str(result_dir), self.url, scenario, self.port)}
        for step in ("prepare", "run"):
            r = subprocess.run(
                ["python3", "-m", "infx.serving.client", step],
                cwd=b.infx_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=b.max_duration_s + 900,
            )
            (result_dir / f"{step}.log").write_text(r.stdout + "\n" + r.stderr)
            if r.returncode != 0:
                return {"ok": False, "error": f"infx client {step} exited {r.returncode}"}
        export = result_dir / "aiperf_artifacts" / "profile_export.jsonl"
        if not export.exists():
            return {"ok": False, "error": f"missing {export}"}
        return parse_profile_export(str(export), self.slo)

    def measure(self, cfg: Dict[str, Any], round_name: str, repeat: int, requests: Optional[int] = None) -> Dict[str, Any]:
        """Restart the server on ``cfg`` and bench it (one process per measurement, as the
        plan requires). Failures are returned as ``ok: False`` records, never raised, so a
        config that cannot start (e.g. OOM) is recorded and the search continues."""
        self._counter += 1
        tag = f"{self._counter:03d}-{round_name}-r{repeat}"
        t = time.time()
        try:
            self.start_server(cfg, self.out_dir / f"server-{tag}.log")
            m = self.bench(requests, tag)
        except Exception as e:  # noqa: BLE001 - recorded in results.jsonl
            m = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.stop_server()
        m["wall_s"] = time.time() - t
        return m
