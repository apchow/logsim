# API Reference (`FastAPI`)

Start the API server locally:

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI docs are available at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service health check and active domain metadata |
| `GET` | `/policies` | List all registered policy IDs (`current_replay`, `fifo_edf`, `nearest_resource`, `fixed_batch`, `greedy_cost`, `ortools_optimizer`, `ml_risk_scorer`, `ppo_agent`) |
| `GET` | `/scenarios` | List versioned scenario definitions loaded from `configs/scenarios.yaml` |
| `POST` | `/datasets/synthetic` | Generate, validate, and persist a deterministic synthetic reverse-logistics snapshot |
| `POST` | `/simulate` | Run a single `(policy_id, scenario_id, seed)` episode and store results in DuckDB + Parquet |
| `POST` | `/backtest` | Execute a multi-seed paired backtest grid and return summary, statistical uplift, and CSV/Parquet/HTML/PDF paths |
| `GET` | `/runs` | Query recorded experiment runs from DuckDB |
| `GET` | `/runs/{run_id}/events` | Retrieve the immutable time-ordered event stream for a run |
| `POST` | `/analyze` | Generate on-demand Gemini API (`gemini-4-hydrogen`) executive comparison summary |
| `POST` | `/routing/matrix` | Compute or replay a deterministic cached OSRM / GraphHopper / Planar route matrix |

## Internal MCP Server (`logisticsbt-mcp`)

Run `python -m app.mcp_server.server` (or `logisticsbt-mcp`) to expose 9 typed tools over MCP 2.x stdio:
`run_backtest`, `get_backtest_results`, `compare_policies`, `optimize_routes`, `get_route_matrix`, `query_events`, `explain_decision`, `create_scenario`, `validate_dataset`.
