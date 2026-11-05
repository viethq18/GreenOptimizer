import pytest

from green_optimizer.objective import LOADED_QUEUE, Shape, auto_objective, rank_key
from green_optimizer.spec import SLO

SLO_ = SLO(ttft_p90_ms=1000, user_tok_s_p90_min=50)


@pytest.mark.parametrize(
    "shape,mode",
    [
        (Shape(queue_len=5.0, ttft_ms=100, decode_ms=10, slo_feasible=False), "goodput"),  # infeasibility wins over load
        (Shape(queue_len=LOADED_QUEUE, ttft_ms=900, decode_ms=10), "balanced"),  # threshold is inclusive
        (Shape(queue_len=0.1, ttft_ms=900, decode_ms=300), "ttft"),
        (Shape(queue_len=0.1, ttft_ms=200, decode_ms=3000), "itl"),
    ],
)
def test_auto_objective_branches(shape, mode):
    assert auto_objective(shape)[0] == mode


def test_rank_key_rejects_unknown_and_auto():
    m = {"ok": True, "ttft_p90_ms": 1, "tok_s_slow": 100, "throughput_tok_s": 1}
    for bad in ("auto", "latency"):
        with pytest.raises(ValueError):
            rank_key(bad, SLO_, m)
