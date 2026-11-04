"""Roofline + closed-loop queueing model of TTFT, per-user tok/s and memory feasibility.

The model is deliberately small: every term is a byte or FLOP count divided by a hardware
peak and an efficiency ``eta`` that is *calibrated* against measured runs
(``configs/calibration/*.yaml``, see ``Calibration``). It is used only to prefilter the
configuration space and to rank candidates; real numbers always come from ``runner.py``.

Terms
-----
prefill   t(new, prefix) = (2*P*new + 393216*new*(prefix + new/2)) / (peak_fp8 * eta_p)
          (393216 = 4*q_heads*head_dim*full_attn_layers FLOPs per (q,k) pair)
decode    step(B) = (W + W_draft + B*(kv_bytes_per_tok*ctx/eta_kv + 2*S_gdn)) / (BW * eta_d)
          S_gdn is read AND written every step; kv read efficiency depends on the attention
          backend (fa4 0.79 measured, green 0.85 = A2 acceptance target, not yet measured).
spec      accept(D) = sum_{i<D} p^i (geometric acceptance); a verify step costs
          (1 + v*(D-1)) of a plain step, normalised so the calibrated D=4 point is exact.
memory    rest = mem_fraction*HBM - weights - draft - reserved; mamba gets ratio/(1+ratio)
          of it (sglang kv_cache_configurator); KV gets the remainder.
queue     one prefill server (M/G/1, Pollaczek-Khinchine wait) fed by a closed loop of
          ``concurrency`` lanes with think time Z, solved by bisection on the throughput.
          TTFT p90 = prefill_p90 + mean_wait * tail_factor * ln(10 rho)/rho.
tok/s     fast tail = accept/step, slow tail (10th percentile, the SLO side) =
          accept/(step + phi*rho*chunk_time) because decode batches are interleaved with
          prefill chunks of ``chunked_prefill_size`` tokens.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .spec import (
    SPEC_OFF,
    Hardware,
    LogNormal,
    ModelSpec,
    Workload,
)

GIB = 2**30
QUANTILES = 512


class CalibParams(BaseModel):
    """Efficiencies/fudge factors fitted on measured runs (provenance in the yaml)."""

    model_config = ConfigDict(extra="forbid")

    eta_p: float = Field(gt=0, le=1)  # prefill MFU vs the dense fp8 peak
    eta_d: float = Field(gt=0, le=1)  # decode HBM efficiency (absorbs draft model, launch gaps)
    kv_read_efficiency: Dict[str, float]  # per attention backend
    mean_stall_fraction: float = Field(ge=0)  # mean prefill-chunk share per decode step (see decode_state)
    kv_usable_fraction: float = Field(gt=0, le=1)  # share of KV tokens usable by running reqs
    slow_tail_prefill_fraction: float = Field(ge=0)  # phi
    ttft_tail_factor: float = Field(gt=0)
    mamba_derate: float = Field(gt=0, le=1)  # realised / nominal mamba share
    reserved_gib: float = Field(ge=0)  # activations, cuda graphs, fragmentation
    spec_accept_p: float = Field(gt=0, lt=1)
    verify_overhead_per_token: float = Field(ge=0)  # v above
    calibrated_draft_tokens: int = 4
    # prefill efficiency of a chunk of c tokens ~ c/(c+chunk_half_tokens), normalised to the
    # calibrated chunk size. ASSUMPTION (only one chunk size was measured): it only supplies the
    # direction of the chunk-size trade-off (small chunk: less decode stall, slower prefill).
    calibrated_chunk_size: int = 8192
    chunk_half_tokens: float = 1024.0


class CalibRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str  # "fitted" (parameters were tuned on it) or "held_out"
    config: Dict[str, Any]
    workload: Dict[str, Any]  # Workload fields
    observed: Dict[str, float]


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provenance: str
    fitted_parameters: List[str]
    predicted_metrics: List[str]
    caveat: str
    max_rel_error: float = 0.20
    params: CalibParams
    runs: List[CalibRun]


def load_calibration(path) -> Calibration:
    with open(path) as f:
        return Calibration.model_validate(yaml.safe_load(f))


def resolve_calibration(hw: Hardware, hw_path) -> Optional[Calibration]:
    if hw.calibration is None:
        return None
    return load_calibration(Path(hw_path).parent / hw.calibration)


class Prediction(BaseModel):
    feasible: bool
    reason: str = ""
    lam: float = 0.0  # requests / s
    total_tok_s: float = 0.0  # (ISL+OSL) * lam, the throughput metric of the plan baseline
    ttft_mean_ms: float = math.inf
    ttft_p90_ms: float = math.inf
    tok_s_fast: float = 0.0
    tok_s_slow: float = 0.0
    step_ms: float = 0.0
    accept: float = 1.0
    running: float = 0.0
    utilization: float = 0.0
    kv_tokens: int = 0
    mamba_slots: int = 0
    max_running_effective: int = 0
    in_system: float = 0.0


def _chunk_factor(chunk: float, p: CalibParams) -> float:
    return chunk / (chunk + p.chunk_half_tokens)


def spec_tokens(cfg: Dict[str, Any]) -> int:
    d = cfg["speculative_num_draft_tokens"]
    return 1 if d == SPEC_OFF else int(d)


def accept_length(cfg: Dict[str, Any], p: CalibParams) -> float:
    d = spec_tokens(cfg)
    if d == 1:
        return 1.0
    return sum(p.spec_accept_p**i for i in range(d))


def memory_plan(cfg, hw: Hardware, model: ModelSpec, p: CalibParams) -> Dict[str, float]:
    spec_on = cfg["speculative_num_draft_tokens"] != SPEC_OFF
    draft = model.draft_weight_gib if spec_on else 0.0
    rest = cfg.get("mem_fraction_static", 0.85) * hw.hbm_gib - model.weight_gib_resident - draft - p.reserved_gib
    rest_b = max(rest, 0.0) * GIB
    ratio = float(cfg["mamba_full_memory_ratio"])
    mamba_b = rest_b * ratio / (1.0 + ratio) * p.mamba_derate
    slot_b = model.gdn_state_bytes(cfg["mamba_ssm_dtype"]) + model.gdn_extra_bytes_per_slot
    slots = int(mamba_b // slot_b)
    kv_b = rest_b - mamba_b
    per_tok = model.kv_bytes_per_token(cfg["kv_cache_dtype"]) + (model.draft_kv_bytes_per_token if spec_on else 0.0)
    return {
        "rest_gib": rest,
        "mamba_slots": slots,
        "kv_tokens": int(kv_b // per_tok),
        "max_requests_by_mamba": slots // model.mamba_slots_per_request,
    }


def prefill_time(new: float, prefix: float, hw: Hardware, model: ModelSpec, eta_p: float) -> float:
    flops = 2.0 * model.active_params * new
    flops += 4.0 * model.q_heads * model.head_dim * model.full_attn_layers * new * (prefix + new / 2.0)
    return flops / (hw.peak_fp8_tflops * 1e12 * eta_p)


def _decode_step(batch: float, ctx: float, cfg, hw, model, p: CalibParams) -> float:
    spec_on = cfg["speculative_num_draft_tokens"] != SPEC_OFF
    w = model.weight_bytes + (model.draft_weight_gib * GIB if spec_on else 0.0)
    kv_eff = p.kv_read_efficiency[cfg["attention_backend"]]
    kv = model.kv_bytes_per_token(cfg["kv_cache_dtype"]) * ctx / kv_eff
    gdn = 2.0 * model.gdn_state_bytes(cfg["mamba_ssm_dtype"])
    t = (w + batch * (kv + gdn)) / (hw.hbm_bw * p.eta_d)
    d = spec_tokens(cfg)
    v = p.verify_overhead_per_token
    return t * (1.0 + v * (d - 1)) / (1.0 + v * (p.calibrated_draft_tokens - 1))


class _Dist:
    """ISL/OSL sample sets of a workload for one prefix-hit rate."""

    def __init__(self, wl: Workload, hit: float):
        self.isl = wl.isl.quantiles(QUANTILES)
        self.osl = wl.osl.quantiles(QUANTILES)
        self.hit = hit
        self.isl_mean = sum(self.isl) / len(self.isl)
        self.osl_mean = sum(self.osl) / len(self.osl)


def predict(cfg: Dict[str, Any], wl: Workload, hw: Hardware, model: ModelSpec, p: CalibParams) -> Prediction:
    """Predict steady-state metrics of ``cfg`` under ``wl``; never raises for a bad config."""
    mem = memory_plan(cfg, hw, model, p)
    if mem["rest_gib"] <= 0 or mem["kv_tokens"] <= 0:
        return Prediction(feasible=False, reason="no HBM left for KV/mamba cache")
    cap = min(int(cfg["max_running_requests"]), mem["max_requests_by_mamba"])
    if cap < 1:
        return Prediction(feasible=False, reason="mamba pool holds < 1 request")

    hit = wl.hit_rate(bool(cfg["hicache"]))
    dist = _Dist(wl, hit)
    ctx = dist.isl_mean + dist.osl_mean / 2.0
    cap = min(cap, max(1, int(p.kv_usable_fraction * mem["kv_tokens"] / ctx)))
    if p.kv_usable_fraction * mem["kv_tokens"] < dist.isl_mean:
        return Prediction(feasible=False, reason="KV pool smaller than one mean-length request")

    news = [x * (1.0 - hit) for x in dist.isl]
    eta_pe = p.eta_p * _chunk_factor(float(cfg["chunked_prefill_size"]), p) / _chunk_factor(p.calibrated_chunk_size, p)
    pre = [prefill_time(n, x - n, hw, model, eta_pe) for n, x in zip(news, dist.isl)]
    sp = sum(pre) / len(pre)
    sp2 = sum(t * t for t in pre) / len(pre)
    pre_sorted = sorted(pre)
    pre_p90 = pre_sorted[int(0.9 * (len(pre) - 1))]
    acc = accept_length(cfg, p)
    steps = dist.osl_mean / acc
    chunk_tokens = min(float(cfg["chunked_prefill_size"]), sum(news) / len(news))
    chunk_t = prefill_time(chunk_tokens, ctx, hw, model, eta_pe)

    def decode_state(lam: float):
        """Running batch by Little's law: B = lam * decode wall time per request.

        The scheduler runs a prefill chunk whenever the queue is non-empty, so every decode
        step of a request is preceded, on average, by ``mean_stall_fraction * rho_p`` of a
        chunk (the slow tail uses ``slow_tail_prefill_fraction``)."""
        stall = p.mean_stall_fraction * min(1.0, lam * sp) * chunk_t
        b = 1.0
        for _ in range(30):
            step = _decode_step(b, ctx, cfg, hw, model, p)
            nb = min(max(1.0, lam * steps * (step + stall)), float(cap))
            if abs(nb - b) < 1e-6:
                break
            b = 0.5 * (b + nb)
        step = _decode_step(b, ctx, cfg, hw, model, p)
        return b, step, steps * (step + stall)

    def service(lam: float):
        b, step, wall = decode_state(lam)
        gpu_dec = steps * step / b  # GPU time of decode per request (shared by b requests)
        return b, step, wall, gpu_dec

    def wait(lam: float):
        _, _, _, gpu_dec = service(lam)
        rho = lam * (sp + gpu_dec)
        if rho >= 1.0:
            return rho, math.inf
        es2 = sp2 + 2.0 * sp * gpu_dec + gpu_dec**2
        return rho, lam * es2 / (2.0 * (1.0 - rho))

    if wl.arrival_rate is not None:
        lam = wl.arrival_rate
        rho, wq = wait(lam)
        if rho >= 1.0:
            return Prediction(feasible=False, reason=f"open loop overload: rho={rho:.2f}", lam=lam, utilization=rho)
    else:
        z = wl.think_time_s
        conc = float(wl.concurrency)
        lo, hi = 0.0, 1.0 / sp  # rho <= 1 implies lam <= 1/sp; rho >= 1 gives infinite wait
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            b, step, wall, _ = service(mid)
            rho_m, wq_m = wait(mid)
            resp = wq_m + sp + wall
            if mid * (z + resp) < conc:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        rho, wq = wait(lam)

    b, step, wall, _ = service(lam)
    rho_t = max(rho, 1e-9)
    tail = max(math.log(10.0 * rho_t) / rho_t, 0.0)
    ttft_mean = wq + sp
    ttft_p90 = pre_p90 + wq * p.ttft_tail_factor * tail
    fast = acc / step
    slow = acc / (step + p.slow_tail_prefill_fraction * min(1.0, rho) * chunk_t)
    return Prediction(
        feasible=True,
        lam=lam,
        total_tok_s=lam * (dist.isl_mean + dist.osl_mean),
        ttft_mean_ms=ttft_mean * 1e3,
        ttft_p90_ms=ttft_p90 * 1e3,
        tok_s_fast=fast,
        tok_s_slow=slow,
        step_ms=step * 1e3,
        accept=acc,
        running=b,
        utilization=rho,
        kv_tokens=mem["kv_tokens"],
        mamba_slots=mem["mamba_slots"],
        max_running_effective=min(int(cfg["max_running_requests"]), mem["max_requests_by_mamba"]),
        in_system=(float(wl.concurrency) - lam * wl.think_time_s) if wl.concurrency else lam * (ttft_mean + wall),
    )


def ttft_lower_bound(wl: Workload, hw: Hardware, model: ModelSpec, p: Optional[CalibParams], hit: float = 0.0) -> Dict[str, float]:
    """Unloaded TTFT of a mean-length request: physical (eta=1) and calibrated bound."""
    isl = wl.isl.mean()
    new = isl * (1.0 - hit)
    out = {
        "isl_mean": isl,
        "hit": hit,
        "physical_s": prefill_time(new, isl - new, hw, model, 1.0),
    }
    if p is not None:
        out["calibrated_s"] = prefill_time(new, isl - new, hw, model, p.eta_p)
    return out
