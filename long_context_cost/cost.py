"""Deterministic long-context inference cost estimates."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    parameter_billion: float
    hidden_size: int
    layers: int
    kv_heads: int
    bytes_per_value: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    gpu_hour_cost_usd: float


@dataclass(frozen=True)
class Workload:
    input_tokens: int
    output_tokens: int
    requests: int


def estimate_serving_cost(model: ModelProfile, workload: Workload) -> dict[str, float]:
    if workload.input_tokens < 0 or workload.output_tokens < 0 or workload.requests < 0:
        raise ValueError("token and request counts must be non-negative")
    kv_cache_gb = _kv_cache_gb(model, workload.input_tokens + workload.output_tokens)
    prefill_seconds = workload.requests * workload.input_tokens / max(model.prefill_tokens_per_second, 1e-9)
    decode_seconds = workload.requests * workload.output_tokens / max(model.decode_tokens_per_second, 1e-9)
    gpu_hours = (prefill_seconds + decode_seconds) / 3600.0
    return {
        "kv_cache_gb": round(kv_cache_gb, 3),
        "prefill_seconds": round(prefill_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "gpu_hours": round(gpu_hours, 6),
        "cost_usd": round(gpu_hours * model.gpu_hour_cost_usd, 4),
    }


def _kv_cache_gb(model: ModelProfile, total_tokens: int) -> float:
    values = 2 * model.layers * model.kv_heads * total_tokens * model.hidden_size
    return values * model.bytes_per_value / 1_000_000_000


def sample_model() -> ModelProfile:
    return ModelProfile("open-1m-context", 120.0, 128, 80, 16, 2, 150_000.0, 2200.0, 3.8)
