"""FastAPI service exposing reverse-logistics simulation, backtesting, routing, and registry APIs."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException

from app.api.schemas import (
    AnalyzeRequest,
    BacktestRequest,
    BacktestResponse,
    OptimizeRoutesRequest,
    RouteMatrixRequest,
    ScenarioCreateRequest,
    SimulateRequest,
    SyntheticDatasetRequest,
    ValidateDatasetRequest,
)
from app.config import load_app_settings
from app.data.normalization import save_dataset_snapshot
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_dataset
from app.evaluation.ai_analyst import analyze_experiment_with_gemini
from app.experiments.registry import ExperimentRegistry
from app.experiments.scenarios import load_scenarios
from app.mcp_server.tools import (
    tool_compare_policies,
    tool_create_scenario,
    tool_explain_decision,
    tool_get_route_matrix,
    tool_optimize_routes,
    tool_query_events,
    tool_run_backtest,
    tool_simulate_once,
    tool_validate_dataset,
)
from app.policies.registry import available_policy_ids

app = FastAPI(
    title="Logistics Backtester API",
    version="0.1.0",
    description="Discrete-event backtesting API for reverse-logistics AI, OR-Tools, and rule policies.",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness health check."""
    return {"status": "ok", "domain": "reverse_pickup_and_hub_triage"}


@app.get("/policies")
def list_policies() -> dict[str, list[str]]:
    """List all registered decision policies."""
    return {"policies": available_policy_ids()}


@app.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    """List versioned scenarios configured in scenarios.yaml and configs/scenarios/."""
    scenarios = load_scenarios()
    return [s.model_dump(mode="json") for s in scenarios.values()]


@app.post("/scenarios")
def create_scenario(req: ScenarioCreateRequest) -> dict[str, Any]:
    """Validate and persist a versioned scenario overlay config under configs/scenarios/."""
    try:
        return tool_create_scenario(
            scenario_id=req.scenario_id,
            name=req.name,
            description=req.description,
            overlays=req.overlays,
            horizon_minutes=req.horizon_minutes,
            version=req.version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/datasets/synthetic")
def create_synthetic_dataset(req: SyntheticDatasetRequest) -> dict[str, Any]:
    """Generate, validate, and save a synthetic reverse-logistics snapshot."""
    ds = generate_synthetic_dataset(
        seed=req.seed,
        horizon_minutes=req.horizon_minutes,
        orders_per_day=req.orders_per_day,
        dataset_id=req.dataset_id,
        routing_provider=req.routing_provider,
    )
    report = validate_dataset(ds)
    out = save_dataset_snapshot(ds, req.output_dir)
    return {
        "dataset_id": ds.dataset_id,
        "snapshot_id": ds.snapshot_id,
        "is_synthetic": ds.is_synthetic,
        "snapshot_path": str(out),
        "validation": report.model_dump(mode="json"),
    }


@app.post("/datasets/validate")
def validate_dataset_endpoint(
    dataset_path: str | None = None,
    req: Annotated[ValidateDatasetRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Validate a dataset snapshot, SHA-256 manifest, dep directory bundle, or CSV/Parquet file."""
    target = dataset_path or (req.dataset_path if req is not None else "data/synthetic/default")
    try:
        return tool_validate_dataset(target)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/simulate")
def simulate_once(req: SimulateRequest) -> dict[str, Any]:
    """Run a single simulation episode and record to registry."""
    try:
        return tool_simulate_once(
            scenario_id=req.scenario_id,
            policy_id=req.policy_id,
            seed=req.seed,
            dataset_path=req.dataset_path,
            experiment_name=req.experiment_name,
            routing_provider=req.routing_provider,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Run multi-seed paired backtest across scenarios and policies."""
    try:
        out = tool_run_backtest(
            scenario_ids=req.scenario_ids,
            policy_ids=req.policy_ids,
            seeds=req.seeds,
            experiment_name=req.experiment_name,
            baseline_policy_id=req.baseline_policy_id,
            dataset_path=req.dataset_path,
            routing_provider=req.routing_provider,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BacktestResponse(
        experiment_name=out["experiment_name"],
        run_count=out["run_count"],
        csv_path=out["csv_path"],
        parquet_path=out["parquet_path"],
        html_path=out["html_path"],
        pdf_path=out["pdf_path"],
        summary=out["summary"],
        comparison=out["comparison"],
    )


@app.get("/runs")
def get_runs(experiment_name: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """List recorded runs from the DuckDB experiment registry."""
    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    return registry.list_runs(experiment_name=experiment_name, limit=limit)


@app.get("/runs/{run_id}/events")
def get_run_events(run_id: str, limit: int = 300) -> list[dict[str, Any]]:
    """Fetch immutable time-ordered event stream for a specific run."""
    res = tool_query_events(run_id=run_id, limit=limit)
    return list(res["events"])


@app.get("/runs/{run_id}/explain")
def explain_run_decision(run_id: str, decision_id: str | None = None) -> dict[str, Any]:
    """Return cost/bottleneck attribution and causal observation_hash traces for a recorded run."""
    try:
        return tool_explain_decision(run_id=run_id, decision_id=decision_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/experiments/{experiment_name}/compare")
def compare_experiment_policies(
    experiment_name: str,
    baseline_policy_id: str = "current_replay",
    practical_threshold_pct: float = 3.0,
) -> dict[str, Any]:
    """Return paired statistical comparison & effect sizes for an experiment."""
    return tool_compare_policies(
        experiment_name=experiment_name,
        baseline_policy_id=baseline_policy_id,
        practical_threshold_pct=practical_threshold_pct,
    )


@app.post("/optimize/routes")
def optimize_routes_endpoint(req: OptimizeRoutesRequest) -> dict[str, Any]:
    """Run constrained Google OR-Tools CP-SAT dispatch assignment over candidate orders and vehicles."""
    return tool_optimize_routes(
        orders=req.orders,
        vehicles=req.vehicles,
        timestamp_minute=req.timestamp_minute,
        solver_time_limit_sec=req.solver_time_limit_sec,
    )


@app.post("/analyze")
def analyze_experiment(req: AnalyzeRequest) -> dict[str, Any]:
    """Generate an on-demand Gemini AI executive comparison summary for an experiment."""
    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    df = registry.get_kpi_dataframe(experiment_name=req.experiment_name)
    if df.empty:
        raise HTTPException(
            status_code=404, detail=f"Experiment '{req.experiment_name}' not found"
        )
    return analyze_experiment_with_gemini(
        experiment_name=req.experiment_name,
        exp_df=df,
        baseline_policy_id=req.baseline_policy_id,
        question=req.question,
        model=req.model,
    )


@app.post("/routing/matrix")
def compute_route_matrix(req: RouteMatrixRequest) -> dict[str, Any]:
    """Compute or replay a deterministic cached OSRM / GraphHopper / planar route matrix."""
    return tool_get_route_matrix(
        origins=req.origins,
        destinations=req.destinations,
        profile=req.profile,
        provider=req.provider,
        departure_minute=req.departure_minute,
    )
