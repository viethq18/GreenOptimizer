"""Online-knob handlers and admission AROUND hook, driven by a fake scheduler (no SGLang objects)."""

import pytest

from green_optimizer import plugin


class Rejected(Exception):
    pass


class Sched:
    def __init__(self, chunk=8192, cap=51):
        self.chunked_prefill_size = chunk
        self.max_running_requests = cap


def _knobs(s):
    return plugin.OnlineKnobs(s, Rejected)


def test_chunk_range_and_apply():
    s = Sched()
    k = _knobs(s)
    k.validate_chunk(plugin.CHUNK_KEY, 4096)
    assert k.apply_chunk(plugin.CHUNK_KEY, 4096) is None
    assert s.chunked_prefill_size == 4096
    # the ceiling stays the startup value, not the lowered one
    k.validate_chunk(plugin.CHUNK_KEY, 8192)
    for bad in (512, 8193, "4096", 4096.0, True):
        with pytest.raises(Rejected):
            k.validate_chunk(plugin.CHUNK_KEY, bad)


def test_chunk_rejected_without_chunked_prefill():
    with pytest.raises(Rejected):
        _knobs(Sched(chunk=None)).validate_chunk(plugin.CHUNK_KEY, 2048)


def test_cap_range_and_apply():
    s = Sched()
    k = _knobs(s)
    k.validate_cap(plugin.CAP_KEY, 16)
    assert k.apply_cap(plugin.CAP_KEY, 16) is None
    assert s._green_max_running_cap == 16
    for bad in (0, 52, "8", None):
        with pytest.raises(Rejected):
            k.validate_cap(plugin.CAP_KEY, bad)


def test_admit_around_lowers_and_restores_limit():
    s = Sched()
    s._green_max_running_cap = 16
    seen = []

    def orig(self, *a, **k):
        seen.append(self.max_running_requests)
        return "batch"

    assert plugin.admit_around(orig, s) == "batch"
    assert seen == [16] and s.max_running_requests == 51


def test_admit_around_restores_on_error_and_passes_through_without_cap():
    s = Sched()
    s._green_max_running_cap = 8

    def boom(self):
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        plugin.admit_around(boom, s)
    assert s.max_running_requests == 51
    s2 = Sched()
    seen = []
    plugin.admit_around(lambda self: seen.append(self.max_running_requests), s2)
    assert seen == [51]
