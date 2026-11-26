# Logistics Backtester

Production-ready MVP platform for backtesting AI, optimization, and rule-based operational policies on discrete-event **reverse logistics** simulations (return pickup dispatching & hub triage).

## Capabilities

- **Discrete-Event Engine (`SimPy`)**: Models reverse pickup vehicles, multi-stop routes, driver Hours-of-Service (HOS), hub dock bays, inspection/grading pickers, sortation robots, quarantine/restock/refurbishment inventory, stockouts, and 20 immutable time-ordered event types.
- **Zero Data Leakage**: Policies observe only `DecisionContext` / Gymnasium observations constructed strictly from events at `t <= sim_time`, excluding ground-truth dispositions and future cancellations/disruptions.
- **Hard Operational Constraints**: `ConstraintGuard` blocks capacity (kg, m3, units), hazmat safety segregation, breakdown, shift calendar, HOS, and hub storage violations while reporting violation counts.
- **Common Gymnasium Interface**: Every policy (`current_replay`, `do_nothing`, `fifo_edf`, `nearest_resource`, `fixed_batch`, `greedy_cost`, `ortools_optimizer`, `ml_risk_scorer`, `ppo_agent`) executes identically via `reset()`, `step(action)`, `observation_space`, and `action_space`.
- **Reproducible Experiment Registry**: Stores `config_hash`, Git commit, `data_snapshot_id`, seed, policy version, scenario ID, KPI metrics, runtime, DuckDB tables, Parquet event logs, and MLflow tracking runs.
- **Paired Multi-Seed Evaluation**: Reports mean, median, P5, P95, std, paired t-test, Wilcoxon, permutation p-values, Holm-Bonferroni correction, Student's t & paired bootstrap 95% confidence intervals, Cohen's d effect sizes, constraint violations, and per-scenario uplift with CSV/Parquet/HTML/PDF exports.
- **Fail-Closed Data Contracts & Atomic Snapshots**: Preserves string IDs (`"0001"`, `"NA"`), rejects duplicate headers and mixed synthetic/historical provenance, supports multi-file bundles (`examples/demo/`) and directory scenarios (`configs/scenarios/*.yaml`), and publishes SHA-256 verified snapshots.
- **Deterministic Cached Routing & Internal MCP Server**: Exposes OSRM, GraphHopper, and Planar distance/duration matrix adapters with persistent SHA-256 response caching (`app/routing/`), on-demand Gemini API executive summaries (`POST /analyze`), and an internal 9-tool MCP server (`logisticsbt-mcp`).

## Quickstart

```bash
uv sync --extra dev
uv run logisticsbt ingest --synthetic --seed 42
uv run logisticsbt validate
uv run logisticsbt train --timesteps 15000
uv run logisticsbt backtest
uv run logisticsbt report
```

Launch UI & API:

```bash
uv run python -m app.dashboard.gradio_app
uv run streamlit run app/dashboard/main.py
uv run uvicorn app.api.main:app --port 8000
```

See [docs/architecture.md](docs/architecture.md), [docs/data-contract.md](docs/data-contract.md), [docs/schema.sql](docs/schema.sql), [docs/api.md](docs/api.md), and [docs/demo_walkthrough.md](docs/demo_walkthrough.md).
