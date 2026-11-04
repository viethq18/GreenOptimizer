"""SGLang plugin entry point: online knobs for the GreenOptimizer sidecar (O2) and the
green-srpt prefill scheduler.

Adds two keys to ``/set_internal_state`` (``{"server_args": {...}}``):

``green_chunked_prefill_size``  int in [1024, startup chunked_prefill_size]
    Assigned to ``scheduler.chunked_prefill_size``. The scheduler reads it live at every
    use (``get_new_batch_prefill``, the mixed-chunk and prefill-delayer paths); the only
    copy taken at init is ``DynamicChunkSizer`` which exists only with pp_size > 1 and
    ``--enable-dynamic-chunking`` (not a supported configuration of this plugin).
``green_max_running_cap``  int in [1, startup max_running_requests]
    Admission cap. ``Scheduler.get_new_batch_prefill`` hands ``self.max_running_requests``
    to the prefill adder; an AROUND hook lowers that attribute to the cap for the duration
    of the call and restores it in ``finally`` (the scheduler loop is single-threaded, and
    nothing else observes the attribute while a batch is being formed). Requests already
    running are never evicted: the cap only stops new admissions.

green-srpt (``GREEN_SRPT=1`` with ``--schedule-policy shortest-prefill-first``): short
uncached prefills go first so they do not queue behind long prompts, with token aging
so long prompts cannot starve.

* Order: SGLang's shortest-prefill-first sort key ``uncached work`` becomes
  ``uncached work - age * waited tokens``. ``waited tokens`` is the prefill tokens the
  scheduler processed since the request was queued (the counter HRRN uses). A request of
  work ``w`` stays behind fresh requests of work ``s`` for at most ``(w - s) / age``
  prefill tokens.
* Chunk sharing: while a long prompt is being chunked, its chunk shrinks to admit the
  waiting requests that score below its remaining work in the same batch (SGLang's
  ``shortest_prefill_chunk_limit``), but it keeps ``min_share`` of the budget every round.
* Online keys, only registered with ``GREEN_SRPT=1``: ``green_srpt_age`` (float in
  [0, 1]) and ``green_srpt_min_share`` (float in [0.05, 1]; 1 disables chunk sharing).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

logger = logging.getLogger(__name__)

CHUNK_KEY = "green_chunked_prefill_size"
CAP_KEY = "green_max_running_cap"
MIN_CHUNK = 1024

SRPT_ENV = "GREEN_SRPT"
SRPT_POLICY = "green-srpt"  # GreenOptimizer's schedule_policy value
SRPT_BASE_POLICY = "shortest-prefill-first"  # the SGLang policy green-srpt extends
AGE_KEY = "green_srpt_age"
SHARE_KEY = "green_srpt_min_share"
MIN_SHARE = 0.05

_POLICY = "sglang.srt.managers.schedule_policy.SchedulePolicy"
PRIORITY_TARGET = _POLICY + ".calc_priority"
SORT_TARGET = _POLICY + "._sort_by_shortest_prefill"
LIMIT_TARGET = _POLICY + ".shortest_prefill_chunk_limit"

HANDLERS_TARGET = (
    "sglang.srt.managers.scheduler_components.internal_state."
    "SchedulerInternalStateController._update_handlers"
)
ADMIT_TARGET = "sglang.srt.managers.scheduler.Scheduler.get_new_batch_prefill"

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/plugins/hook_registry.py": ("HookRegistry", "HookType", "register", "AFTER", "AROUND"),
    "srt/managers/scheduler_components/internal_state.py": (
        "SchedulerInternalStateController",
        "_update_handlers",
        "_UpdateHandler",
        "UpdateRejected",
        "set_internal_state(recv_req)",
    ),
    "srt/managers/scheduler.py": (
        "get_new_batch_prefill",
        "chunked_prefill_size",
        "max_running_requests",
        "arrival_processed_tokens",
    ),
    "srt/managers/schedule_policy.py": (
        "SchedulePolicy",
        "calc_priority(processed_tokens)",
        "_sort_by_shortest_prefill(temporary_deprioritized)",
        "shortest_prefill_chunk_limit(page_size)",
        "policy",
        '"shortest-prefill-first"',
    ),
    "srt/managers/schedule_batch.py": (
        "rid",
        "origin_input_ids",
        "output_ids",
        "num_matched_prefix_tokens",
        "full_untruncated_fill_ids",
        "prefix_len",
        "time_stats",
    ),
    "srt/observability/req_time_stats.py": ("wait_queue_entry_time",),
}


def _as_int(key: str, value: Any, reject) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise reject(f"Updating {key} to {value!r} is rejected: expected an integer.")
    return value


def _as_float(key: str, value: Any, lo: float, hi: float, reject) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise reject(f"Updating {key} to {value!r} is rejected: expected a number.")
    if not lo <= value <= hi:
        raise reject(f"Updating {key} to {value} is rejected: valid range is [{lo}, {hi}].")
    return float(value)


def srpt_enabled() -> bool:
    return os.environ.get(SRPT_ENV) == "1"


class SrptState:
    """green-srpt state. Process-wide: each scheduler (TP/DP rank) is its own process."""

    def __init__(self, age: float = 0.25, min_share: float = 0.25):
        self.age = age
        self.min_share = min_share
        self.processed = 0  # SGLang's prefill-token counter at the last calc_priority
        self.deprioritized: FrozenSet[str] = frozenset()  # rids SGLang deferred this round


SRPT = SrptState()


def _ceil_pages(n: int, page_size: int) -> int:
    return -(-n // page_size) * page_size


def _work(r) -> int:
    """Uncached prefill tokens; SGLang's ``SchedulePolicy._shortest_prefill_work``."""
    return max(1, len(r.origin_input_ids) + len(r.output_ids) - r.num_matched_prefix_tokens)


def srpt_score(r, processed: int, age: float) -> float:
    waited = max(0, processed - r.arrival_processed_tokens)
    return _work(r) - age * waited


def srpt_sort(waiting_queue: List, deprioritized: Iterable[str], processed: int, age: float) -> None:
    deferred = frozenset(deprioritized)
    waiting_queue.sort(
        key=lambda r: (
            r.rid in deferred,
            srpt_score(r, processed, age),
            r.time_stats.wait_queue_entry_time,
        )
    )


def srpt_chunk_limit(
    chunked_req,
    waiting_queue: List,
    deprioritized: FrozenSet[str],
    budget: int,
    page_size: int,
    processed: int,
    age: float,
    min_share: float,
) -> Optional[int]:
    """Chunk cap for the request being chunked: reserve budget for the waiting requests
    (sorted by ``srpt_sort``) that score below its remaining work, keeping at least
    ``min_share`` of the budget for it. ``None`` = no cap (SGLang's contract)."""
    if budget < 2 * page_size:
        return None
    remaining = len(chunked_req.full_untruncated_fill_ids) - chunked_req.prefix_len
    room = budget - max(page_size, _ceil_pages(int(budget * min_share), page_size))
    reserved = 0
    for r in waiting_queue:
        if r.rid in deprioritized or srpt_score(r, processed, age) >= remaining:
            break
        charge = _ceil_pages(_work(r), page_size)
        if reserved + charge > room:
            break
        reserved += charge
    if not reserved:
        return None
    # Page alignment keeps continuation boundaries allocator-compatible.
    return (budget - reserved) // page_size * page_size


def _policy_name(policy) -> str:
    return getattr(policy, "value", policy)


def priority_around(original_fn, self, waiting_queue, running_batch=None, processed_tokens=0, *rest, **kwargs):
    """AROUND hook for SchedulePolicy.calc_priority: refuse a wrong base policy, record the
    prefill-token counter for the sort and chunk-limit hooks."""
    if _policy_name(self.policy) != SRPT_BASE_POLICY:
        raise RuntimeError(
            f"{SRPT_ENV}=1 needs --schedule-policy {SRPT_BASE_POLICY}, "
            f"got {_policy_name(self.policy)}"
        )
    SRPT.processed = processed_tokens
    return original_fn(self, waiting_queue, running_batch, processed_tokens, *rest, **kwargs)


def sort_around(original_fn, waiting_queue, temporary_deprioritized):
    """AROUND hook for SchedulePolicy._sort_by_shortest_prefill: aged SRPT order."""
    SRPT.deprioritized = frozenset(temporary_deprioritized)
    srpt_sort(waiting_queue, SRPT.deprioritized, SRPT.processed, SRPT.age)


def limit_around(original_fn, self, chunked_req, waiting_queue, budget, page_size):
    """AROUND hook for SchedulePolicy.shortest_prefill_chunk_limit."""
    return srpt_chunk_limit(
        chunked_req,
        waiting_queue,
        SRPT.deprioritized,
        budget,
        page_size,
        SRPT.processed,
        SRPT.age,
        SRPT.min_share,
    )


class OnlineKnobs:
    """Validation/apply logic, independent of SGLang objects (``reject`` builds the
    exception, so unit tests can drive it with a fake scheduler)."""

    def __init__(self, scheduler, reject):
        self.scheduler = scheduler
        self.reject = reject

    def _startup_chunk(self) -> int:
        s = self.scheduler
        if not hasattr(s, "_green_startup_chunked"):
            s._green_startup_chunked = s.chunked_prefill_size
        return s._green_startup_chunked

    def validate_chunk(self, key: str, value: Any) -> None:
        v = _as_int(key, value, self.reject)
        top = self._startup_chunk()
        if top is None:
            raise self.reject(f"Updating {key} is rejected: the server runs without chunked prefill.")
        if not MIN_CHUNK <= v <= top:
            raise self.reject(f"Updating {key} to {v} is rejected: valid range is [{MIN_CHUNK}, {top}].")

    def apply_chunk(self, key: str, value: Any):
        self.scheduler.chunked_prefill_size = int(value)
        logger.info("green_optimizer: chunked_prefill_size -> %d", value)
        return None  # a non-empty dict would be forwarded to the config override

    def validate_cap(self, key: str, value: Any) -> None:
        v = _as_int(key, value, self.reject)
        top = self.scheduler.max_running_requests
        if not 1 <= v <= top:
            raise self.reject(f"Updating {key} to {v} is rejected: valid range is [1, {top}].")

    def apply_cap(self, key: str, value: Any):
        self.scheduler._green_max_running_cap = int(value)
        logger.info("green_optimizer: admission cap -> %d", value)
        return None

    def validate_age(self, key: str, value: Any) -> None:
        _as_float(key, value, 0.0, 1.0, self.reject)

    def apply_age(self, key: str, value: Any):
        SRPT.age = float(value)
        logger.info("green_optimizer: srpt age -> %g", value)
        return None

    def validate_share(self, key: str, value: Any) -> None:
        _as_float(key, value, MIN_SHARE, 1.0, self.reject)

    def apply_share(self, key: str, value: Any):
        SRPT.min_share = float(value)
        logger.info("green_optimizer: srpt min_share -> %g", value)
        return None


def admit_around(original_fn, self, *args, **kwargs):
    """AROUND hook for Scheduler.get_new_batch_prefill: enforce the admission cap."""
    cap = getattr(self, "_green_max_running_cap", None)
    if cap is None or cap >= self.max_running_requests:
        return original_fn(self, *args, **kwargs)
    full = self.max_running_requests
    self.max_running_requests = cap
    try:
        return original_fn(self, *args, **kwargs)
    finally:
        self.max_running_requests = full


def add_handlers(result: Dict[str, Any], controller) -> Dict[str, Any]:
    """AFTER hook for SchedulerInternalStateController._update_handlers."""
    from sglang.srt.managers.scheduler_components.internal_state import (
        UpdateRejected,
        _UpdateHandler,
    )

    knobs = getattr(controller, "_green_knobs", None)
    if knobs is None:
        knobs = controller._green_knobs = OnlineKnobs(controller.scheduler, UpdateRejected)
    result[CHUNK_KEY] = _UpdateHandler(knobs.validate_chunk, knobs.apply_chunk)
    result[CAP_KEY] = _UpdateHandler(knobs.validate_cap, knobs.apply_cap)
    if srpt_enabled():
        result[AGE_KEY] = _UpdateHandler(knobs.validate_age, knobs.apply_age)
        result[SHARE_KEY] = _UpdateHandler(knobs.validate_share, knobs.apply_share)
    return result


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(HANDLERS_TARGET, add_handlers, HookType.AFTER)
    HookRegistry.register(ADMIT_TARGET, admit_around, HookType.AROUND)
    if srpt_enabled():
        HookRegistry.register(PRIORITY_TARGET, priority_around, HookType.AROUND)
        HookRegistry.register(SORT_TARGET, sort_around, HookType.AROUND)
        HookRegistry.register(LIMIT_TARGET, limit_around, HookType.AROUND)
        logger.info("green_optimizer: green-srpt scheduling on (age=%g, min_share=%g)", SRPT.age, SRPT.min_share)
