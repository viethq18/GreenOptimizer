"""Typed inputs of GreenOptimizer: workload, SLO, hardware, model, search space.

Everything is loaded from YAML (``load_*``) and validated with pydantic, so a typo in a
config fails at load time, not after a 20 minute server launch.

SLO percentile convention
-------------------------
The plan states the SLO "at p90, measured on the slow side". For TTFT that is the 90th
percentile (large = bad). For per-user decode speed large = good, so the slow side is the
*low* tail: ``user_tok_s_p90_min`` is the speed that 90% of requests meet or beat, i.e.
the 10th percentile of the per-user tok/s distribution. aiperf's
``output_token_throughput_per_user`` "p90" is the *fast* tail (68.6 tok/s in the native
livekit run) and is NOT what this SLO constrains; the slow tail of that run is 5.57 tok/s.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterator, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# SLO slow-tail percentile of the per-user tok/s distribution (see module docstring)
TOK_S_SLOW_PERCENTILE = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LogNormal(_Strict):
    """Token-length distribution: ln X ~ N(mu, sigma), clipped to [min, max]."""

    mu: float
    sigma: float = Field(ge=0)
    min: float = 1
    max: float = 1e9

    def quantiles(self, n: int = 2048) -> List[float]:
        """n equally likely samples (midpoint quantiles), clipped; deterministic."""
        if self.sigma == 0:
            return [min(max(math.exp(self.mu), self.min), self.max)] * n
        nd = NormalDist(self.mu, self.sigma)
        return [
            min(max(math.exp(nd.inv_cdf((i + 0.5) / n)), self.min), self.max)
            for i in range(n)
        ]

    def mean(self) -> float:
        q = self.quantiles()
        return sum(q) / len(q)


class BenchSpec(_Strict):
    """How runner.py drives the load generator (InferenceX ``infx.serving.client``)."""

    scenario_yaml: str  # aiperf scenario, relative to infx_dir
    dataset: str  # SERVING_DATASET
    scenario_label: str  # SERVING_SCENARIO_LABEL
    dataset_kind: str = "file"  # SERVING_DATASET_KIND
    infx_dir: str = "/network-volume/rnd-sglang/InferenceX-wt-livekit/inferencex-e2e"
    aiperf_runtime_dir: str = "/network-volume/rnd-sglang/.aiperf-runtime-livekit"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    # synthetic scenarios: dataset isl/osl are rewritten into the scenario yaml
    synthetic_isl: Optional[int] = None
    synthetic_osl: Optional[int] = None
    # requests/duration of the three halving rounds are set by search.py; this is the
    # wall-clock cap (s) of one measurement so a stuck server cannot hang the search
    max_duration_s: float = 1800.0


class Workload(_Strict):
    name: str
    dataset: str
    # closed loop (lanes recycle) when concurrency is set, open loop when arrival_rate is
    concurrency: Optional[int] = Field(default=None, ge=1)
    arrival_rate: Optional[float] = Field(default=None, gt=0)  # requests / s
    isl: LogNormal
    osl: LogNormal
    prefix_hit_rate: float = Field(ge=0, lt=1)  # share of ISL served from the prefix cache
    # hit rate when the hierarchical (host) cache is off; None = same as prefix_hit_rate
    prefix_hit_rate_no_hicache: Optional[float] = Field(default=None, ge=0, lt=1)
    think_time_s: float = Field(default=0.0, ge=0)  # mean gap between a lane's turns
    bench: Optional[BenchSpec] = None
    notes: str = ""

    @model_validator(mode="after")
    def _one_load_model(self) -> "Workload":
        if (self.concurrency is None) == (self.arrival_rate is None):
            raise ValueError("set exactly one of concurrency (closed loop) / arrival_rate (open loop)")
        return self

    def hit_rate(self, hicache: bool) -> float:
        if hicache or self.prefix_hit_rate_no_hicache is None:
            return self.prefix_hit_rate
        return self.prefix_hit_rate_no_hicache


class SLO(_Strict):
    ttft_p90_ms: float = Field(gt=0)
    # per-user decode speed that 90% of requests meet or beat (aiperf p10 of
    # output_token_throughput_per_user); see module docstring
    user_tok_s_p90_min: float = Field(gt=0)

    def violation(self, ttft_p90_ms: float, tok_s_slow: float) -> float:
        """Worst relative violation: 0 when met, 0.25 when 25% off the limit."""
        v_ttft = max(0.0, ttft_p90_ms / self.ttft_p90_ms - 1.0)
        v_tok = max(0.0, self.user_tok_s_p90_min / max(tok_s_slow, 1e-9) - 1.0)
        return max(v_ttft, v_tok)


class Hardware(_Strict):
    name: str
    sm_count: int
    hbm_gib: float
    hbm_bw_tbs: float  # TB/s
    peak_fp8_tflops: float  # dense
    peak_bf16_tflops: float  # dense
    calibration: Optional[str] = None  # path relative to this file

    @property
    def hbm_bytes(self) -> float:
        return self.hbm_gib * 2**30

    @property
    def hbm_bw(self) -> float:
        return self.hbm_bw_tbs * 1e12


class ModelSpec(_Strict):
    """Qwen3.8-27B-FP8 hybrid (config.json of /network-volume/rnd-sglang/models/)."""

    name: str = "Qwen3.8-27B-FP8"
    weight_bytes: float = 27e9  # FP8 weights read once per decode step
    weight_gib_resident: float = 28.39  # server_native.log "weights" (incl. scales/embeds)
    active_params: float = 27e9  # 2 FLOPs per param per token
    full_attn_layers: int = 16
    gdn_layers: int = 48
    q_heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    # GDN recurrent state, fp32: 141.5 MiB per slot (server_native.log: 21.94 GiB / 155 slots)
    gdn_state_bytes_fp32: float = 148.4e6
    # conv state + replayssm intermediate window per slot (0.43 + 0.29 GiB / 155 slots)
    gdn_extra_bytes_per_slot: float = 4.9e6
    mamba_slots_per_request: int = 3
    draft_weight_gib: float = 3.69  # DFlash2 draft model
    draft_kv_bytes_per_token: float = 19.0e3  # bf16 K+V of the draft, per token
    # decode: a 1-token step reads weights; draft model is read once per step when spec is on
    max_context: int = 256000

    def kv_bytes_per_token(self, kv_dtype: str) -> float:
        elem = 1 if kv_dtype.startswith("fp8") else 2
        return 2 * self.full_attn_layers * self.kv_heads * self.head_dim * elem

    def gdn_state_bytes(self, ssm_dtype: str) -> float:
        return self.gdn_state_bytes_fp32 * (0.5 if ssm_dtype == "bfloat16" else 1.0)


# --------------------------------------------------------------------------- search space
SPEC_OFF = "off"

# flag -> candidate values (plan L433-440). mem-fraction-static is fixed, not searched.
DEFAULT_SPACE: Dict[str, List[Any]] = {
    "attention_backend": ["fa4", "green"],
    "kv_cache_dtype": ["fp8_e4m3", "bfloat16"],
    "chunked_prefill_size": [4096, 8192, 16384],
    "max_prefill_tokens": [8192, 16384, 32768],
    "max_running_requests": [16, 32, 48, 64, 128],
    "mamba_ssm_dtype": ["float32", "bfloat16"],
    "mamba_full_memory_ratio": [3, 5, 8],
    "speculative_num_draft_tokens": [SPEC_OFF, 2, 4, 8],
    "schedule_policy": ["dfs-weight", "lpm", "fcfs"],
    "hicache": [True, False],
}

# knobs that can change while the server runs (O2 ladder); everything else needs a restart
ONLINE_KNOBS = ("chunked_prefill_size", "max_running_requests")


class SearchSpace(_Strict):
    space: Dict[str, List[Any]] = Field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SPACE.items()})
    fixed: Dict[str, Any] = Field(default_factory=lambda: {"mem_fraction_static": 0.85})

    @model_validator(mode="after")
    def _known(self) -> "SearchSpace":
        unknown = set(self.space) - set(DEFAULT_SPACE)
        if unknown:
            raise ValueError(f"unknown search-space knob(s): {sorted(unknown)}")
        for k, vals in self.space.items():
            if not vals:
                raise ValueError(f"search-space knob {k} has no candidate values")
        return self

    def size(self) -> int:
        n = 1
        for vals in self.space.values():
            n *= len(vals)
        return n

    def configs(self) -> Iterator[Dict[str, Any]]:
        """Every combination; unsearched knobs take the first DEFAULT_SPACE value."""
        keys = list(DEFAULT_SPACE)
        pools = [self.space.get(k, DEFAULT_SPACE[k][:1]) for k in keys]
        for combo in itertools.product(*pools):
            cfg = dict(zip(keys, combo))
            cfg.update(self.fixed)
            yield cfg


# --------------------------------------------------------------------------- loading
def _load_yaml(path: Union[str, Path]) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_workload(path: Union[str, Path]) -> Workload:
    return Workload.model_validate(_load_yaml(path))


def load_slo(path: Union[str, Path]) -> SLO:
    return SLO.model_validate(_load_yaml(path))


def load_hardware(path: Union[str, Path]) -> Hardware:
    return Hardware.model_validate(_load_yaml(path))


def load_space(path: Union[str, Path]) -> SearchSpace:
    return SearchSpace.model_validate(_load_yaml(path))


def config_key(cfg: Dict[str, Any]) -> str:
    """Stable identity of a config (used for results.jsonl and de-duplication)."""
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))
