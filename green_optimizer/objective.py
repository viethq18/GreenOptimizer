"""What "best" means: objective modes shared by the offline search and the online controller.

Every mode ranks a metrics mapping (a roofline prediction or a measurement, keys
``ok, ttft_p90_ms, tok_s_slow, throughput_tok_s`` and optionally ``goodput_req_s``):

  balanced    meet the SLO, then maximise total tok/s (the original ranking)
  ttft        meet the SLO, then minimise TTFT p90
  itl         meet the SLO, then maximise slow-tail per-user tok/s (1 / ITL)
  goodput     maximise requests/s that individually meet both SLO thresholds
  throughput  maximise total tok/s, SLO ignored (batch / offline serving)
  auto        pick one of the above from the workload's shape (``auto_objective``)

Nothing here depends on a model: the inputs are queueing and latency quantities that
every model produces, so the same rules hold for any ``ModelSpec``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .spec import SLO

OBJECTIVES = ("balanced", "ttft", "itl", "goodput", "throughput")
MODES = OBJECTIVES + ("auto",)
# objectives whose ranking puts SLO-meeting configs first (goodput embeds the SLO per
# request, throughput ignores it)
SLO_TIERED = ("balanced", "ttft", "itl")

# auto: mean number of waiting requests at which the server counts as loaded. Below it
# capacity is idle and is better spent on latency; at or above it, queueing dominates.
LOADED_QUEUE = 0.5


def rank_key(objective: str, slo: SLO, m: Mapping[str, Any], slack: float = 0.0) -> Tuple[int, float]:
    """Sort key (smaller is better). Tier 0 = acceptable under ``objective``, tier 1 =
    best effort ordered by SLO violation, tier 2 = unusable."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    if not m.get("ok", False):
        return (2, math.inf)
    v = slo.violation(m["ttft_p90_ms"], m["tok_s_slow"])
    if objective == "throughput":
        return (0, -m["throughput_tok_s"])
    if objective == "goodput":
        g = m.get("goodput_req_s")
        return (0, -g) if g else (1, v)
    if v > slack:
        return (1, v)
    if objective == "balanced":
        return (0, -m["throughput_tok_s"])
    if objective == "ttft":
        return (0, m["ttft_p90_ms"])
    return (0, -m["tok_s_slow"])  # itl


@dataclass
class Shape:
    """What the workload looks like to the server, from a prediction or from live metrics.

    queue_len   mean number of requests waiting for their first prefill
    ttft_ms     mean time to first token
    decode_ms   mean decode time of one request (OSL x inter-token latency)
    slo_feasible  some configuration meets the SLO (search) / the SLO is not violated on
                  both axes at once (live)
    """

    queue_len: float
    ttft_ms: float
    decode_ms: float
    slo_feasible: bool = True


def auto_objective(s: Shape) -> Tuple[str, str]:
    """(objective, reason) for a workload shape. Never returns ``throughput``: auto keeps
    the SLO in the objective."""
    if not s.slo_feasible:
        return "goodput", "the p90 SLO is out of reach, so maximise the requests that do meet it"
    if s.queue_len >= LOADED_QUEUE:
        return "balanced", f"loaded (mean queue {s.queue_len:.2f} >= {LOADED_QUEUE}): keep throughput headroom within the SLO"
    if s.ttft_ms >= s.decode_ms:
        return "ttft", f"idle capacity and time to first token dominates latency ({s.ttft_ms:,.0f} ms vs decode {s.decode_ms:,.0f} ms)"
    return "itl", f"idle capacity and decode dominates latency ({s.decode_ms:,.0f} ms vs first token {s.ttft_ms:,.0f} ms)"
