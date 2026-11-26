# API Reference (`FastAPI` & Internal MCP Server)

Start the API server locally:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Interactive OpenAPI docs are available at `http://127.0.0.1:8000/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check and active domain metadata |
| `GET` | `/policies` | List all registered policy IDs (`current_replay`, `fifo_edf`, `nearest_resource`, `fixed_batch`, `greedy_cost`, `ortools_optimizer`, `ml_risk_scorer`, `ppo_agent`, `do_nothing`) |
| `GET` | `/scenarios` | List versioned scenario definitions loaded from `configs/scenarios.yaml` and `configs/scenarios/*.yaml` |
| `POST` | `/scenarios` | Validate and persist a new versioned scenario overlay YAML under `configs/scenarios/` |
| `POST` | `/datasets/synthetic` | Generate, validate, and persist a deterministic synthetic reverse-logistics snapshot (supports optional `routing_provider`) |
| `POST` | `/datasets/validate` | Validate a dataset snapshot, SHA-256 manifest, multi-table `examples/demo` bundle, or CSV/Parquet orders file |
| `POST` | `/simulate` | Run a single `(policy_id, scenario_id, seed)` episode (supporting `dataset_path` & `routing_provider`) and store results in DuckDB + Parquet |
| `POST` | `/backtest` | Execute a multi-seed paired backtest grid and return summary, statistical uplift, and CSV/Parquet/HTML/PDF paths |
| `GET` | `/runs` | Query recorded experiment runs from DuckDB |
| `GET` | `/runs/{run_id}/events` | Retrieve the NaN-safe immutable time-ordered event stream for a run |
| `GET` | `/runs/{run_id}/explain` | Retrieve cost/bottleneck attribution and causal `observation_hash` decision trace for a recorded run |
| `GET` | `/experiments/{experiment_name}/compare` | Return paired statistical comparison, Holm-Bonferroni adjusted p-values, 95% CIs, and Cohen's d |
| `POST` | `/optimize/routes` | Solve constrained dispatch assignment using Google OR-Tools CP-SAT over candidate orders & vehicles |
| `POST` | `/analyze` | Generate on-demand Gemini API (`gemini-4-hydrogen`) executive comparison summary with 8s timeout & deterministic fallback |
| `POST` | `/routing/matrix` | Compute or replay a deterministic cached OSRM / GraphHopper / Planar route matrix |

## Internal MCP Server (`logisticsbt mcp` / `logisticsbt-mcp`)

Run `uv run logisticsbt mcp` (or `python -m app.mcp_server.server`) to expose 9 typed tools over MCP 2.x stdio backed by the same shared handlers:
- `run_backtest`
- `get_backtest_results`
- `compare_policies`
- `optimize_routes`
- `get_route_matrix`
- `query_events`
- `explain_decision`
- `create_scenario`
- `validate_dataset`
