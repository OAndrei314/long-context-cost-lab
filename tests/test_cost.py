import pytest

from long_context_cost import ModelProfile, Workload, estimate_serving_cost


def test_cost_increases_with_context_length():
    model = ModelProfile("test", 30.0, 64, 32, 8, 2, 100_000.0, 2000.0, 2.0)
    short = estimate_serving_cost(model, Workload(8_000, 512, 1))
    long = estimate_serving_cost(model, Workload(1_000_000, 512, 1))

    assert long["kv_cache_gb"] > short["kv_cache_gb"]
    assert long["cost_usd"] > short["cost_usd"]


def test_negative_workload_rejected():
    model = ModelProfile("test", 30.0, 64, 32, 8, 2, 100_000.0, 2000.0, 2.0)

    with pytest.raises(ValueError):
        estimate_serving_cost(model, Workload(-1, 0, 1))
