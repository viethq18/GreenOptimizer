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


class AttnGroup(_Strict):
    """Layers sharing one attention shape. ``window`` = sliding window in tokens (None = full)."""

    name: str
    layers: int = Field(gt=0)
    q_heads: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    window: Optional[int] = Field(default=None, gt=0)

    def kv_bytes_per_token(self, kv_dtype: str) -> float:
        """K + V bytes of one token over all layers of the group."""
        elem = 1 if kv_dtype.startswith("fp8") else 2
        return 2 * self.layers * self.kv_heads * self.head_dim * elem


class GdnSpec(_Strict):
    """Linear-attention (GDN / mamba) state held in sglang's mamba pool."""

    layers: int = Field(gt=0)
    state_bytes_fp32: float = Field(gt=0)  # recurrent state per slot, fp32
    extra_bytes_per_slot: float = Field(ge=0)  # conv state + replay window per slot
    slots_per_request: int = Field(gt=0)

    def state_bytes(self, ssm_dtype: str) -> float:
        return self.state_bytes_fp32 * (0.5 if ssm_dtype == "bfloat16" else 1.0)


class DraftSpec(_Strict):
    """Speculative-decoding draft model (read once per step when spec is on)."""

    weight_gib: float = Field(gt=0)
    kv_bytes_per_token: float = Field(ge=0)


class Unsupported(_Strict):
    """Knob combination that cannot run for this model: ``when`` matches a config if every
    listed knob takes one of the listed values. ``predict`` reports it infeasible with ``reason``."""

    when: Dict[str, List[Any]]
    reason: str

    def matches(self, cfg: Dict[str, Any]) -> bool:
        return all(cfg.get(k) in vals for k, vals in self.when.items())


class ModelSpec(_Strict):
    """Served model, loaded from ``configs/models/*.yaml`` (provenance comments live there)."""

    name: str
    served_name: str  # must equal the launch template's --served-model-name
    weight_bytes: float = Field(gt=0)  # bytes read once per decode step
    weight_gib_resident: float = Field(gt=0)  # sglang "Load weight end ... mem usage"
    active_params: float = Field(gt=0)  # 2 FLOPs per param per token
    max_context: int = Field(gt=0)
    attention: List[AttnGroup] = Field(min_length=1)
    gdn: Optional[GdnSpec] = None
    draft: Optional[DraftSpec] = None
    unsupported: List[Unsupported] = Field(default_factory=list)

    @property
    def full_groups(self) -> List[AttnGroup]:
        return [g for g in self.attention if g.window is None]

    @property
    def swa_groups(self) -> List[AttnGroup]:
        return [g for g in self.attention if g.window is not None]

    @property
    def window(self) -> Optional[int]:
        """Sliding window of the SWA pool (sglang has one window per model)."""
        ws = {g.window for g in self.swa_groups}
        return ws.pop() if ws else None

    @model_validator(mode="after")
    def _one_window(self) -> "ModelSpec":
        if len({g.window for g in self.swa_groups}) > 1:
            raise ValueError("sglang's hybrid SWA pool supports one sliding window per model")
        if not self.full_groups:
            raise ValueError("at least one full-attention group is required (sglang sizes the pool by it)")
        return self

    def full_kv_bytes_per_token(self, kv_dtype: str) -> float:
        return sum(g.kv_bytes_per_token(kv_dtype) for g in self.full_groups)

    def swa_kv_bytes_per_token(self, kv_dtype: str) -> float:
        return sum(g.kv_bytes_per_token(kv_dtype) for g in self.swa_groups)

    def unsupported_reason(self, cfg: Dict[str, Any]) -> Optional[str]:
        for u in self.unsupported:
            if u.matches(cfg):
                return u.reason
        return None


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
    # SWA pool tokens / full pool tokens (sglang --swa-full-tokens-ratio, default 0.8)
    "swa_full_tokens_ratio": [0.8, 0.4, 0.2, 0.1, 0.05],
    "speculative_num_draft_tokens": [SPEC_OFF, 2, 4, 8],
    # opt-in extra value: "green-srpt" (shortest-prefill-first + aging + chunk sharing, plugin.py),
    # list it in a --space yaml to search it
    "schedule_policy": ["dfs-weight", "lpm", "fcfs"],
    "hicache": [True, False],
}

# Opt-in knobs: searched only when listed in a --space yaml; otherwise absent from every config,
# so no flag reaches the server and the default search is unchanged. First value = sglang's default.
OPT_IN_SPACE: Dict[str, List[Any]] = {
    # --enable-mixed-chunk: decode requests ride along in prefill-chunk batches
    "enable_mixed_chunk": [False, True],
    # --prefill-decode-interval: decode rounds after a prefill batch before the next one
    # (None = flag not passed, sglang's per-model default)
    "prefill_decode_interval": [None, 4, 8, 16],
    # --enable-prefill-delayer with --prefill-delayer-queue-min-ratio (None = delayer off)
    "prefill_delayer_queue_min_ratio": [None, 0.1, 0.3, 0.5],
}

# sglang's value when --swa-full-tokens-ratio is not given (server_args.py)
SWA_FULL_TOKENS_RATIO_DEFAULT = 0.8

# knobs that can change while the server runs (O2 ladder); everything else needs a restart
ONLINE_KNOBS = ("chunked_prefill_size", "max_running_requests")


def inapplicable_knobs(model: "ModelSpec") -> Dict[str, str]:
    """Knobs the model has no use for -> why. They are left out of every config, so no flag
    is passed to the server for them."""
    out: Dict[str, str] = {}
    if model.gdn is None:
        out["mamba_ssm_dtype"] = out["mamba_full_memory_ratio"] = "no GDN/mamba layers"
    if not model.swa_groups:
        out["swa_full_tokens_ratio"] = "no sliding-window layers"
    if model.draft is None:
        out["speculative_num_draft_tokens"] = "no draft model configured (speculative decoding off)"
    return out


class SearchSpace(_Strict):
    space: Dict[str, List[Any]] = Field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_SPACE.items()})
    fixed: Dict[str, Any] = Field(default_factory=lambda: {"mem_fraction_static": 0.85})
    # knob -> reason; set by for_model(), excluded from configs()
    excluded: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known(self) -> "SearchSpace":
        unknown = (set(self.space) | set(self.excluded)) - set(DEFAULT_SPACE) - set(OPT_IN_SPACE)
        if unknown:
            raise ValueError(f"unknown search-space knob(s): {sorted(unknown)}")
        for k, vals in self.space.items():
            if not vals:
                raise ValueError(f"search-space knob {k} has no candidate values")
        both = set(self.space) & set(self.excluded)
        if both:
            raise ValueError(f"knob(s) both searched and excluded: {sorted(both)}")
        return self

    def for_model(self, model: "ModelSpec") -> "SearchSpace":
        """Copy without the knobs ``model`` cannot use (see ``inapplicable_knobs``)."""
        drop = {**self.excluded, **inapplicable_knobs(model)}
        return SearchSpace(space={k: v for k, v in self.space.items() if k not in drop}, fixed=dict(self.fixed), excluded=drop)

    def size(self) -> int:
        n = 1
        for vals in self.space.values():
            n *= len(vals)
        return n

    def configs(self) -> Iterator[Dict[str, Any]]:
        """Every combination; unsearched knobs take the first DEFAULT_SPACE value, excluded
        knobs and unlisted ``OPT_IN_SPACE`` knobs are absent."""
        keys = [k for k in DEFAULT_SPACE if k not in self.excluded] + [k for k in OPT_IN_SPACE if k in self.space]
        pools = [self.space[k] if k in self.space else DEFAULT_SPACE[k][:1] for k in keys]
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


def load_model(path: Union[str, Path]) -> ModelSpec:
    return ModelSpec.model_validate(_load_yaml(path))


def load_space(path: Union[str, Path]) -> SearchSpace:
    return SearchSpace.model_validate(_load_yaml(path))


def config_key(cfg: Dict[str, Any]) -> str:
    """Stable identity of a config (used for results.jsonl and de-duplication)."""
    return ",".join(f"{k}={cfg[k]}" for k in sorted(cfg))
