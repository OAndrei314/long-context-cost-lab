import pytest

from long_context_cost import HardwareProfile, ModelProfile, Workload, compare_context_windows, estimate_serving_cost, memory_feasibility_report


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


def test_memory_feasibility_reports_required_gpu_count():
    model = ModelProfile("test", 120.0, 128, 80, 16, 2, 100_000.0, 2000.0, 2.0)
    workload = Workload(1_000_000, 4096, 1)
    report = memory_feasibility_report(model, workload, HardwareProfile("small", 80.0, 2))

    assert report["decision"] == "needs_more_gpus_or_quantization"
    assert report["required_gpu_count"] > 2
    assert report["total_memory_gb"] > report["usable_memory_gb"]


def test_compare_context_windows_is_monotonic_for_kv_memory():
    model = ModelProfile("test", 30.0, 64, 32, 8, 2, 100_000.0, 2000.0, 2.0)
    rows = compare_context_windows(model, output_tokens=512, requests=1, context_windows=(8_000, 128_000, 1_000_000))

    assert [row["context_tokens"] for row in rows] == [8000.0, 128000.0, 1000000.0]
    assert rows[0]["kv_cache_gb"] < rows[1]["kv_cache_gb"] < rows[2]["kv_cache_gb"]
