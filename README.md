# Logistics Backtester

Production-ready MVP platform for backtesting AI, optimization, and rule-based operational policies on discrete-event **reverse logistics** simulations (return pickup dispatching & hub triage).

## Capabilities

- **Discrete-Event Engine (`SimPy`)**: Models reverse pickup vehicles, multi-stop pickup tours refined via nearest-neighbor + **2-opt local optimization**, driver Hours-of-Service (HOS), hub dock bays, inspection/grading pickers, sortation robots, quarantine/restock/refurbishment inventory, stockouts, load-weighted emissions, and 20 immutable time-ordered event types.
- **Zero Data Leakage**: Policies observe only `DecisionContext` / Gymnasium observations constructed strictly from events at `t <= sim_time`, excluding ground-truth dispositions and future cancellations/disruptions.
- **Hard Operational Constraints**: `ConstraintGuard` blocks capacity (kg, m3, units), hazmat safety segregation, breakdown, shift calendar, HOS, and hub storage violations while reporting violation counts.
- **Common Gymnasium Interface**: Every policy (`current_replay`, `do_nothing`, `fifo_edf`, `nearest_resource`, `fixed_batch`, `greedy_cost`, `ortools_optimizer`, `ml_risk_scorer`, `ppo_agent`) executes identically via `reset()`, `step(action)`, `observation_space`, and `action_space`.
- **Reproducible Experiment Registry**: Stores `config_hash`, Git commit, `data_snapshot_id`, routing provenance (`routing_provider`, `route_matrix_sha256`), seed, policy version, scenario ID, KPI metrics, runtime, DuckDB tables, Parquet event logs, and MLflow tracking runs.
- **Paired Multi-Seed Evaluation**: Reports mean, median, P5, P95, std, paired t-test, Wilcoxon, permutation p-values, Holm-Bonferroni correction, Student's t & paired bootstrap 95% confidence intervals, Cohen's d effect sizes, constraint violations, and per-scenario uplift with CSV/Parquet/HTML/PDF exports.
- **Fail-Closed Data Contracts & Atomic Snapshots**: Preserves string IDs (`"0001"`, `"NA"`), rejects duplicate headers and mixed synthetic/historical provenance, supports multi-file bundles (`examples/demo/`) and directory scenarios (`configs/scenarios/*.yaml`), and publishes SHA-256 verified snapshots.
- **Deterministic Cached Routing & Internal MCP Server**: Exposes OSRM, GraphHopper, and Planar distance/duration matrix adapters with persistent SHA-256 response caching (`app/routing/`), on-demand Gemini API executive summaries (`POST /analyze` with strict 8s timeout + deterministic fallback), and an internal 9-tool MCP server (`logisticsbt mcp` / `logisticsbt-mcp`).

## Quickstart

```bash
uv sync --extra dev
uv run logisticsbt ingest --synthetic --seed 42
uv run logisticsbt validate --dataset data/synthetic/default
uv run logisticsbt route-matrix --provider osrm
uv run logisticsbt train --timesteps 15000
uv run logisticsbt backtest --routing-provider osrm
uv run logisticsbt report
```

Launch UI, API, and MCP Server:

```bash
uv run python -m app.dashboard.gradio_app   # Fast cached Gradio UI at http://127.0.0.1:7860
uv run uvicorn app.api.main:app --port 8000  # FastAPI REST backend at http://127.0.0.1:8000/docs
uv run logisticsbt mcp                       # Internal MCP 2.x stdio server (9 typed tools)
uv run streamlit run app/dashboard/main.py   # Legacy Streamlit view (optional)
```

### Environment Setup (Gemini API Executive Analyst)

Copy `.env.example` to `.env` (never commit `.env`; it is gitignored) and configure your key:

```bash
GEMINI_API_KEY=your-api-key-here
```

When set, clicking **Analyze Results with Gemini API** in the Gradio dashboard or calling `POST /analyze` invokes `gemini-4-hydrogen` via `google-genai` under an 8s timeout, automatically falling back to deterministic quantitative KPI narration when offline or unconfigured.

See [docs/architecture.md](docs/architecture.md), [docs/data-contract.md](docs/data-contract.md), [docs/schema.sql](docs/schema.sql), [docs/api.md](docs/api.md), and [docs/demo_walkthrough.md](docs/demo_walkthrough.md).
