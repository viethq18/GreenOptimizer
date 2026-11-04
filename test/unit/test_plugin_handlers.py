"""Online-knob handlers, admission AROUND hook and green-srpt scheduling, driven by fakes (no SGLang objects)."""

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


def test_parse_chunk_table_round_trips_and_rejects_bad_tables():
    rows = plugin.parse_chunk_table(" 8:4096, 32:2048 ")
    assert rows == ((8, 4096), (32, 2048))
    assert plugin.parse_chunk_table(plugin.format_chunk_table(rows)) == rows
    assert plugin.parse_chunk_table("") == ()
    for bad in ("8:4096,8:2048", "32:2048,8:4096", "8:2048,32:4096", "0:4096", "8:512", "8", "a:b"):
        with pytest.raises(ValueError):
            plugin.parse_chunk_table(bad)


def test_decode_chunk_cap_takes_the_last_row_reached():
    rows = ((8, 4096), (32, 2048))
    assert [plugin.decode_chunk_cap(rows, b) for b in (0, 7, 8, 31, 32, 99)] == [None, None, 4096, 4096, 2048, 2048]


class Batch:
    def __init__(self, n):
        self.reqs = [object()] * n


def test_admit_around_caps_chunk_by_running_batch_and_restores(monkeypatch):
    monkeypatch.setattr(plugin, "DECODE_CHUNK", ((8, 4096), (32, 2048)))
    s = Sched()
    s._green_max_running_cap = 16
    seen = []

    def orig(self, running_batch):
        seen.append((self.chunked_prefill_size, self.max_running_requests))
        if len(running_batch.reqs) == 40:
            raise RuntimeError("x")
        return "batch"

    assert plugin.admit_around(orig, s, Batch(4)) == "batch"
    assert plugin.admit_around(orig, s, running_batch=Batch(10)) == "batch"
    with pytest.raises(RuntimeError):
        plugin.admit_around(orig, s, Batch(40))
    assert seen == [(8192, 16), (4096, 16), (2048, 16)]
    assert (s.chunked_prefill_size, s.max_running_requests) == (8192, 51)
    s.chunked_prefill_size = 3000  # an online chunk below the table cap wins
    plugin.admit_around(orig, s, Batch(10))
    assert seen[-1][0] == 3000 and s.chunked_prefill_size == 3000


def test_admit_around_chunk_cap_respects_truncation_align_and_disabled_chunking(monkeypatch):
    monkeypatch.setattr(plugin, "DECODE_CHUNK", ((1, 2048),))
    s = Sched()
    s.truncation_align_size = 3072
    seen = []
    plugin.admit_around(lambda self, b: seen.append(self.chunked_prefill_size), s, Batch(4))
    s2 = Sched(chunk=None)
    plugin.admit_around(lambda self, b: seen.append(self.chunked_prefill_size), s2, Batch(4))
    assert seen == [3072, None] and s.chunked_prefill_size == 8192



class Req:
    """The Req fields green-srpt reads: uncached work = input - matched prefix."""

    def __init__(self, rid, n_input, matched=0, arrival=0, entry=0.0, prefix_len=0):
        self.rid = rid
        self.origin_input_ids = [0] * n_input
        self.output_ids = []
        self.num_matched_prefix_tokens = matched
        self.arrival_processed_tokens = arrival
        self.full_untruncated_fill_ids = self.origin_input_ids
        self.prefix_len = prefix_len
        self.time_stats = type("T", (), {"wait_queue_entry_time": entry})()


def _order(queue, processed, age=0.25, deprioritized=()):
    plugin.srpt_sort(queue, deprioritized, processed, age)
    return [r.rid for r in queue]


def test_srpt_puts_short_uncached_work_first():
    # a 30k prompt with a 29k cache hit is short work; a fresh 3k prompt is not
    q = [Req("long", 20000, entry=1), Req("fresh", 3000, entry=2), Req("hit", 30000, matched=29000, entry=3)]
    assert _order(q, processed=0) == ["hit", "fresh", "long"]


def test_srpt_aging_bounds_the_wait_of_long_prompts():
    # work 9000 vs a fresh 1000 at age 0.25: the long one leads after (9000-1000)/0.25 = 32000
    # prefill tokens of waiting
    def queue(processed):
        return [Req("short", 1000, arrival=processed), Req("long", 9000, arrival=0)]

    assert _order(queue(31996), processed=31996) == ["short", "long"]
    assert _order(queue(32004), processed=32004) == ["long", "short"]
    assert _order(queue(10**6), processed=10**6, age=0.0) == ["short", "long"]  # age 0 = plain SPF


def test_srpt_keeps_sglang_deprioritized_requests_last():
    q = [Req("dup", 10, entry=1), Req("big", 5000, entry=2)]
    assert _order(q, processed=0, deprioritized={"dup"}) == ["big", "dup"]


def _limit(queue, chunked, budget=8192, page=16, processed=0, age=0.25, share=0.25, deprioritized=frozenset()):
    plugin.srpt_sort(queue, deprioritized, processed, age)
    return plugin.srpt_chunk_limit(chunked, queue, deprioritized, budget, page, processed, age, share)


def test_chunk_limit_reserves_page_aligned_budget_for_shorter_waiters():
    chunked = Req("c", 40000, prefix_len=8192)  # 31808 tokens left
    # 1000 -> 1008 and 2000 -> 2000 after page rounding
    assert _limit([Req("a", 1000), Req("b", 2000)], chunked) == 8192 - 1008 - 2000


def test_chunk_limit_keeps_min_share_for_the_chunked_prompt():
    chunked = Req("c", 40000)
    waiters = [Req(f"w{i}", 1500) for i in range(8)]  # 1504 each after page rounding
    # room = 8192 - 2048 = 6144 -> four 1504-token waiters fit, the fifth does not
    assert _limit(waiters, chunked) == 8192 - 4 * 1504
    assert _limit([Req(f"w{i}", 1500) for i in range(8)], chunked, share=1.0) is None


def test_chunk_limit_skips_waiters_that_are_not_shorter():
    chunked = Req("c", 3000)
    assert _limit([Req("bigger", 5000)], chunked) is None
    assert _limit([Req("dup", 100)], chunked, deprioritized=frozenset({"dup"})) is None
    assert _limit([Req("a", 100)], chunked, budget=16) is None  # budget below two pages
    # a head waiter that does not fit stops reservation: the ones behind it never jump it
    assert _limit([Req("old", 9000, arrival=0), Req("a", 100, arrival=10**6)], chunked, processed=10**6) is None


def test_srpt_knobs_range_and_apply(monkeypatch):
    monkeypatch.setattr(plugin, "SRPT", plugin.SrptState())
    k = _knobs(Sched())
    k.validate_age(plugin.AGE_KEY, 0.5)
    k.apply_age(plugin.AGE_KEY, 0.5)
    k.validate_share(plugin.SHARE_KEY, 1)
    k.apply_share(plugin.SHARE_KEY, 1)
    assert (plugin.SRPT.age, plugin.SRPT.min_share) == (0.5, 1.0)
    for bad in (-0.1, 1.5, "0.3", True, None):
        with pytest.raises(Rejected):
            k.validate_age(plugin.AGE_KEY, bad)
    for bad in (0.0, 0.04, 1.01, False):
        with pytest.raises(Rejected):
            k.validate_share(plugin.SHARE_KEY, bad)


def test_priority_around_refuses_other_base_policies_and_records_counter(monkeypatch):
    monkeypatch.setattr(plugin, "SRPT", plugin.SrptState())
    policy = type("P", (), {"policy": "dfs-weight"})()
    with pytest.raises(RuntimeError, match="shortest-prefill-first"):
        plugin.priority_around(lambda *a, **k: None, policy, [])
    policy.policy = "shortest-prefill-first"
    seen = []
    plugin.priority_around(lambda *a, **k: seen.append(a[3]), policy, [], None, processed_tokens=777)
    assert seen == [777] and plugin.SRPT.processed == 777
