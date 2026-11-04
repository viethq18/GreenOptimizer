"""Roofline + closed-loop queueing model of TTFT, per-user tok/s and memory feasibility.

The model is deliberately small: every term is a byte or FLOP count divided by a hardware
peak and an efficiency ``eta`` that is *calibrated* against measured runs
(``configs/calibration/*.yaml``, see ``Calibration``). It is used only to prefilter the
configuration space and to rank candidates; real numbers always come from ``runner.py``.

Terms
-----
prefill   t(new, prefix) = (2*P*new + sum_g 4*q_g*hd_g*L_g*A_g(new, prefix)) / (peak_fp8 * eta_p)
          A_g = integral over the new tokens of the keys each one attends: new*(prefix + new/2)
          for full attention, capped at the window W for sliding-window groups.
decode    step(B) = (W + W_draft + B*(sum_g kv_g*min(ctx, W_g)/eta_kv + 2*S_gdn)) / (BW * eta_d)
          S_gdn (0 without GDN layers) is read AND written every step; kv read efficiency depends
          on the attention backend (fa4 0.79 measured, green 0.85 = A2 acceptance target).
spec      accept(D) = sum_{i<D} p^i (geometric acceptance); a verify step costs
          (1 + v*(D-1)) of a plain step, normalised so the calibrated D=4 point is exact.
memory    rest = mem_fraction*HBM - weights - draft - reserved; with GDN layers mamba gets
          ratio/(1+ratio) of it (sglang kv_cache_configurator), KV the remainder. With
          sliding-window layers the KV share is split like sglang's HybridSWAPoolConfigurator:
          full_tokens = kv / (full_cell + r*swa_cell), swa_tokens = r*full_tokens
          (r = swa_full_tokens_ratio). A running request holds ctx full tokens and
          min(ctx, W) + SWA_EVICTION_INTERVAL swa tokens (sglang evicts out-of-window tokens lazily).
queue     one prefill server (M/G/1, Pollaczek-Khinchine wait) fed by a closed loop of
          ``concurrency`` lanes with think time Z, solved by bisection on the throughput.
          TTFT p90 = prefill_p90 + mean_wait * tail_factor * ln(10 rho)/rho.
tok/s     fast tail = accept/step, slow tail (10th percentile, the SLO side) =
          accept/(step + phi*rho*chunk_time) because decode batches are interleaved with
          prefill chunks of ``chunked_prefill_size`` tokens.
chunk cap the decode-aware cap at running batch B is the largest chunk C with
          accept/(step(B) + phi*t(C)) >= the SLO speed (rho = 1); ``decode_chunk_table``.
unsupported  configs listed in the model's ``unsupported`` block are infeasible with its reason.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .spec import (
    SLO,
    SPEC_OFF,
    SWA_FULL_TOKENS_RATIO_DEFAULT,
    Hardware,
    LogNormal,
    ModelSpec,
    Workload,
)
from .plugin import MIN_CHUNK

GIB = 2**30
QUANTILES = 512
# sglang SGLANG_SWA_EVICTION_INTERVAL default: out-of-window tokens are freed every 128 steps
SWA_EVICTION_INTERVAL = 128
# sglang --page-size of the measured launches (fa4_try1.log server_args)
PAGE_SIZE = 1
# decode_chunk_table rounds its caps down to this many tokens
CHUNK_GRANULE = 256


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

    model: str  # ModelSpec.name of the served model the runs measured
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
    swa_tokens: int = 0
    max_running_effective: int = 0
    in_system: float = 0.0
    wait_ms: float = 0.0  # mean queueing delay before the first prefill
    decode_ms: float = 0.0  # mean decode wall time of one request
    slo_attainment: Optional[float] = None  # fraction of requests meeting both SLO thresholds (predict(slo=...))
    goodput_req_s: Optional[float] = None  # lam * slo_attainment


def _chunk_factor(chunk: float, p: CalibParams) -> float:
    return chunk / (chunk + p.chunk_half_tokens)


def spec_on(cfg: Dict[str, Any]) -> bool:
    return cfg.get("speculative_num_draft_tokens", SPEC_OFF) != SPEC_OFF


def spec_tokens(cfg: Dict[str, Any]) -> int:
    return int(cfg["speculative_num_draft_tokens"]) if spec_on(cfg) else 1


def accept_length(cfg: Dict[str, Any], p: CalibParams) -> float:
    d = spec_tokens(cfg)
    if d == 1:
        return 1.0
    return sum(p.spec_accept_p**i for i in range(d))


def memory_plan(cfg, hw: Hardware, model: ModelSpec, p: CalibParams) -> Dict[str, Any]:
    """HBM split. ``max_requests_by_mamba`` is None for models without GDN layers."""
    on = spec_on(cfg) and model.draft is not None
    draft = model.draft.weight_gib if on else 0.0
    rest = cfg.get("mem_fraction_static", 0.85) * hw.hbm_gib - model.weight_gib_resident - draft - p.reserved_gib
    rest_b = max(rest, 0.0) * GIB
    mamba_b, slots, by_mamba = 0.0, 0, None
    if model.gdn is not None:
        ratio = float(cfg["mamba_full_memory_ratio"])
        mamba_b = rest_b * ratio / (1.0 + ratio) * p.mamba_derate
        slot_b = model.gdn.state_bytes(cfg["mamba_ssm_dtype"]) + model.gdn.extra_bytes_per_slot
        slots = int(mamba_b // slot_b)
        by_mamba = slots // model.gdn.slots_per_request
    kv_b = rest_b - mamba_b
    dt = cfg["kv_cache_dtype"]
    swa_ratio = float(cfg.get("swa_full_tokens_ratio", SWA_FULL_TOKENS_RATIO_DEFAULT)) if model.swa_groups else 0.0
    cell = model.full_kv_bytes_per_token(dt) + swa_ratio * model.swa_kv_bytes_per_token(dt)
    cell += model.draft.kv_bytes_per_token if on else 0.0
    full = int(kv_b // cell)
    return {
        "rest_gib": rest,
        "mamba_slots": slots,
        "kv_tokens": full,
        "swa_tokens": int(full * swa_ratio),
        "max_requests_by_mamba": by_mamba,
    }


def attended_keys(new: float, prefix: float, window: Optional[int]) -> float:
    """Sum over ``new`` causal query tokens after ``prefix`` of the keys each attends,
    min(position, window) (continuous approximation)."""
    a, b = prefix, prefix + new
    if window is None or b <= window:
        return (b * b - a * a) / 2.0
    if a >= window:
        return window * new
    return (window * window - a * a) / 2.0 + window * (b - window)


def prefill_time(new: float, prefix: float, hw: Hardware, model: ModelSpec, eta_p: float) -> float:
    flops = 2.0 * model.active_params * new
    for g in model.attention:
        flops += 4.0 * g.q_heads * g.head_dim * g.layers * attended_keys(new, prefix, g.window)
    return flops / (hw.peak_fp8_tflops * 1e12 * eta_p)


def _decode_step(batch: float, ctx: float, cfg, hw, model: ModelSpec, p: CalibParams) -> float:
    on = spec_on(cfg) and model.draft is not None
    w = model.weight_bytes + (model.draft.weight_gib * GIB if on else 0.0)
    kv_eff = p.kv_read_efficiency[cfg["attention_backend"]]
    dt = cfg["kv_cache_dtype"]
    kv = sum(g.kv_bytes_per_token(dt) * (ctx if g.window is None else min(ctx, g.window)) for g in model.attention) / kv_eff
    gdn = 2.0 * model.gdn.state_bytes(cfg["mamba_ssm_dtype"]) if model.gdn is not None else 0.0
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


def _slo_attainment(slo: SLO, pre: List[float], wq: float, rho: float, tail_factor: float, slow: float, fast: float) -> float:
    """Fraction of requests meeting both SLO thresholds, the two assumed independent.

    TTFT of ISL sample i is ``pre[i] + W``, W the M/G/1 wait with P(W > w) = rho * exp(-w / theta);
    theta = wq * tail_factor / rho is the scale that puts W's p90 where ``ttft_p90`` puts it.
    Per-user tok/s has a piecewise-linear CDF through (slow, 0.1) and (fast, 0.9)."""
    t = slo.ttft_p90_ms / 1e3
    r = min(rho, 1.0)
    theta = wq * tail_factor / max(rho, 1e-9)

    def wait_le(w: float) -> float:
        if w < 0:
            return 0.0
        if r <= 0 or theta <= 0:
            return 1.0
        if math.isinf(theta):
            return 1.0 - r
        return 1.0 - r * math.exp(-w / theta)

    a_ttft = sum(wait_le(t - x) for x in pre) / len(pre)
    need = slo.user_tok_s_p90_min
    if fast <= slow:
        a_tok = 1.0 if need <= slow else 0.0
    else:
        a_tok = 1.0 - min(1.0, max(0.0, 0.1 + 0.8 * (need - slow) / (fast - slow)))
    return a_ttft * a_tok


def predict(cfg: Dict[str, Any], wl: Workload, hw: Hardware, model: ModelSpec, p: CalibParams, slo: Optional[SLO] = None) -> Prediction:
    """Predict steady-state metrics of ``cfg`` under ``wl``; never raises for a bad config.
    With ``slo`` the prediction also carries ``slo_attainment`` and ``goodput_req_s``."""
    reason = model.unsupported_reason(cfg)
    if reason is not None:
        return Prediction(feasible=False, reason=f"unsupported for {model.name}: {reason}")
    if spec_on(cfg) and model.draft is None:
        return Prediction(feasible=False, reason=f"speculative decoding needs a draft model; {model.name} has none")
    mem = memory_plan(cfg, hw, model, p)
    if mem["rest_gib"] <= 0 or mem["kv_tokens"] <= 0:
        return Prediction(feasible=False, reason="no HBM left for KV/mamba cache")
    window = model.window
    if window is not None and mem["swa_tokens"] <= window + PAGE_SIZE:
        return Prediction(feasible=False, reason=f"SWA pool {mem['swa_tokens']} tokens <= window + page (sglang validate_swa_pool_size rejects it)")
    by_mamba = mem["max_requests_by_mamba"]
    hard_cap = int(cfg["max_running_requests"]) if by_mamba is None else min(int(cfg["max_running_requests"]), by_mamba)
    if hard_cap < 1:
        return Prediction(feasible=False, reason="mamba pool holds < 1 request")

    hit = wl.hit_rate(bool(cfg["hicache"]))
    dist = _Dist(wl, hit)
    ctx = dist.isl_mean + dist.osl_mean / 2.0
    cap = min(hard_cap, max(1, int(p.kv_usable_fraction * mem["kv_tokens"] / ctx)))
    if p.kv_usable_fraction * mem["kv_tokens"] < dist.isl_mean:
        return Prediction(feasible=False, reason="KV pool smaller than one mean-length request")
    if window is not None:
        # ASSUMPTION: kv_usable_fraction (fitted on a full-attention pool) also holds for the SWA pool
        swa_per_req = min(ctx, window) + SWA_EVICTION_INTERVAL
        cap = min(cap, max(1, int(p.kv_usable_fraction * mem["swa_tokens"] / swa_per_req)))

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
    att = None if slo is None else _slo_attainment(slo, pre, wq, rho, p.ttft_tail_factor, slow, fast)
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
        swa_tokens=mem["swa_tokens"],
        max_running_effective=hard_cap,
        in_system=(float(wl.concurrency) - lam * wl.think_time_s) if wl.concurrency else lam * (ttft_mean + wall),
        wait_ms=wq * 1e3,
        decode_ms=wall * 1e3,
        slo_attainment=att,
        goodput_req_s=None if att is None else lam * att,
    )


def decode_chunk_table(cfg: Dict[str, Any], wl: Workload, hw: Hardware, model: ModelSpec, p: CalibParams, slo: SLO) -> List[Tuple[int, int]]:
    """Rows (B, C) of the plugin's ``GREEN_DECODE_CHUNK``: from B running requests on the chunk
    is capped at C, the largest multiple of CHUNK_GRANULE in [MIN_CHUNK, chunked_prefill_size)
    whose stall keeps ``predict``'s slow-tail per-user speed at the SLO with the prefill server
    busy (rho = 1): accept / (step(B) + phi * t(C)) >= user_tok_s_p90_min. Where even MIN_CHUNK
    misses it the cap is MIN_CHUNK. From the B where decode alone misses the SLO no chunk can
    rescue it, so the table stops (the last cap holds; [] if that is already B = 1).
    Only rows where the cap falls; [] when none is needed."""
    top = int(cfg["chunked_prefill_size"])
    phi = p.slow_tail_prefill_fraction
    if top <= MIN_CHUNK or phi <= 0:
        return []
    dist = _Dist(wl, wl.hit_rate(bool(cfg["hicache"])))
    ctx = dist.isl_mean + dist.osl_mean / 2.0
    acc = accept_length(cfg, p)
    calib_f = _chunk_factor(p.calibrated_chunk_size, p)

    def chunk_t(c: int) -> float:  # predict's chunk_t at chunk size c
        return prefill_time(c, ctx, hw, model, p.eta_p * _chunk_factor(c, p) / calib_f)

    lo_k, hi_k = -(-MIN_CHUNK // CHUNK_GRANULE), (top - 1) // CHUNK_GRANULE  # cap candidates k * granule
    rows: List[Tuple[int, int]] = []
    for b in range(1, int(cfg["max_running_requests"]) + 1):
        stall = (acc / slo.user_tok_s_p90_min - _decode_step(b, ctx, cfg, hw, model, p)) / phi
        if stall <= 0:
            break  # step(B) grows with B: decode alone misses the SLO from here on
        if chunk_t(top) <= stall:
            continue
        lo, hi = lo_k, hi_k  # largest k with chunk_t(k * granule) <= stall, else lo_k
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if chunk_t(mid * CHUNK_GRANULE) <= stall:
                lo = mid
            else:
                hi = mid - 1
        cap = lo * CHUNK_GRANULE
        if not rows or cap < rows[-1][1]:
            rows.append((b, cap))
    return rows


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
