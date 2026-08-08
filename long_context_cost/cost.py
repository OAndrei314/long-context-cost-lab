"""Deterministic long-context inference cost estimates."""
from __future__ import annotations

import math
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


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpu_memory_gb: float
    gpu_count: int
    memory_headroom: float = 0.82


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


def memory_feasibility_report(model: ModelProfile, workload: Workload, hardware: HardwareProfile) -> dict[str, float | str]:
    if hardware.gpu_count <= 0 or hardware.gpu_memory_gb <= 0:
        raise ValueError("hardware must include positive GPU count and memory")
    summary = estimate_serving_cost(model, workload)
    weight_memory_gb = model.parameter_billion * model.bytes_per_value
    total_memory_gb = weight_memory_gb + float(summary["kv_cache_gb"])
    usable_memory_gb = hardware.gpu_count * hardware.gpu_memory_gb * hardware.memory_headroom
    required_gpu_count = math.ceil(total_memory_gb / (hardware.gpu_memory_gb * hardware.memory_headroom))
    decision = "fits_with_headroom" if total_memory_gb <= usable_memory_gb else "needs_more_gpus_or_quantization"
    return {
        "decision": decision,
        "weight_memory_gb": round(weight_memory_gb, 3),
        "kv_cache_gb": summary["kv_cache_gb"],
        "total_memory_gb": round(total_memory_gb, 3),
        "usable_memory_gb": round(usable_memory_gb, 3),
        "memory_per_gpu_gb": round(total_memory_gb / hardware.gpu_count, 3),
        "required_gpu_count": float(required_gpu_count),
    }


def compare_context_windows(
    model: ModelProfile,
    output_tokens: int,
    requests: int,
    context_windows: tuple[int, ...] = (32_000, 128_000, 1_000_000),
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for context_tokens in context_windows:
        summary = estimate_serving_cost(model, Workload(context_tokens, output_tokens, requests))
        rows.append({"context_tokens": float(context_tokens), **summary})
    return rows


def _kv_cache_gb(model: ModelProfile, total_tokens: int) -> float:
    values = 2 * model.layers * model.kv_heads * total_tokens * model.hidden_size
    return values * model.bytes_per_value / 1_000_000_000


def sample_model() -> ModelProfile:
    return ModelProfile("open-1m-context", 120.0, 128, 80, 16, 2, 150_000.0, 2200.0, 3.8)
