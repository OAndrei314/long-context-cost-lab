"""CLI for long-context serving estimates."""
from __future__ import annotations

from .cost import HardwareProfile, Workload, compare_context_windows, estimate_serving_cost, memory_feasibility_report, sample_model


def main(argv: list[str] | None = None) -> int:
    _ = argv
    summary = estimate_serving_cost(sample_model(), Workload(input_tokens=1_000_000, output_tokens=4_096, requests=1))
    for key, value in summary.items():
        print(f"{key}={value}")
    feasibility = memory_feasibility_report(sample_model(), Workload(1_000_000, 4_096, 1), HardwareProfile("8x-192gb", 192.0, 8))
    print(f"decision={feasibility['decision']}")
    print(f"required_gpu_count={feasibility['required_gpu_count']}")
    print("context_window_sweep=" + ",".join(str(int(row["context_tokens"])) for row in compare_context_windows(sample_model(), 4096, 1)))
    return 0
