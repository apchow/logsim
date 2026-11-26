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
