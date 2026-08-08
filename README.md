# long-context-cost-lab

Maintained by: codex-daily-routine

A deterministic estimator for long-context LLM serving economics. It models prefill,
decode, KV-cache memory and approximate GPU-hour cost so open and closed long-context
claims can be compared with engineering constraints instead of vibes.

## Research + Money Thesis

Frontier coding and agent models increasingly advertise 1M-token context windows. The
money question is whether that context can be served at useful latency, memory footprint
and cost. This project gives a transparent baseline estimator for long-context deployment
tradeoffs.

## Quickstart

```powershell
pip install -r requirements-dev.txt
pip install -e .
python -m pytest -q
python -m long_context_cost
```

## Silicon Valley Interview Hook

The `memory_feasibility_report()` API combines parameter memory, KV cache, GPU count and
headroom into an explicit fit/no-fit decision. The `compare_context_windows()` sweep then
shows how 32k, 128k and 1M-token workloads change memory and cost, which is the argument
AI platform teams need before buying or scheduling scarce accelerators.

## Status

MVP: model/workload profiles, KV-cache estimator, token throughput model, memory
feasibility report, context-window sweep and tests. Next steps: add real benchmark
importers and quantization-aware profiles.

## License

MIT - see [LICENSE](LICENSE).
