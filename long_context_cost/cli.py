"""CLI for long-context serving estimates."""
from __future__ import annotations

from .cost import Workload, estimate_serving_cost, sample_model


def main(argv: list[str] | None = None) -> int:
    _ = argv
    summary = estimate_serving_cost(sample_model(), Workload(input_tokens=1_000_000, output_tokens=4_096, requests=1))
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0
