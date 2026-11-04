"""SGLang plugin entry point: online knobs for the GreenOptimizer sidecar (O2).

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
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

CHUNK_KEY = "green_chunked_prefill_size"
CAP_KEY = "green_max_running_cap"
MIN_CHUNK = 1024

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
    ),
}


def _as_int(key: str, value: Any, reject) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise reject(f"Updating {key} to {value!r} is rejected: expected an integer.")
    return value


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
    return result


def register() -> None:
    """SGLang's plugin loader calls this in every process, before any model is built."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(HANDLERS_TARGET, add_handlers, HookType.AFTER)
    HookRegistry.register(ADMIT_TARGET, admit_around, HookType.AROUND)
