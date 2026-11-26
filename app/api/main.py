"""FastAPI service exposing reverse-logistics simulation, backtesting, and registry APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from app.api.schemas import (
    AnalyzeRequest,
    BacktestRequest,
    BacktestResponse,
    RouteMatrixRequest,
    SimulateRequest,
    SyntheticDatasetRequest,
)
from app.config import load_app_settings
from app.data.normalization import load_dataset_snapshot, save_dataset_snapshot
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_dataset
from app.evaluation.ai_analyst import analyze_experiment_with_gemini
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import run_backtest_suite, run_single_episode
from app.experiments.scenarios import load_scenarios
from app.policies.registry import available_policy_ids, build_policy
from app.routing.adapters import GeoPoint, get_route_matrix

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
    """List versioned scenarios configured in scenarios.yaml."""
    scenarios = load_scenarios()
    return [s.model_dump(mode="json") for s in scenarios.values()]


@app.post("/datasets/synthetic")
def create_synthetic_dataset(req: SyntheticDatasetRequest) -> dict[str, Any]:
    """Generate, validate, and save a synthetic reverse-logistics snapshot."""
    ds = generate_synthetic_dataset(
        seed=req.seed,
        horizon_minutes=req.horizon_minutes,
        orders_per_day=req.orders_per_day,
        dataset_id=req.dataset_id,
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


@app.post("/simulate")
def simulate_once(req: SimulateRequest) -> dict[str, Any]:
    """Run a single simulation episode and record to registry."""
    settings = load_app_settings()
    scenarios = load_scenarios()
    if req.scenario_id not in scenarios:
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_id}' not found")
    dataset = (
        load_dataset_snapshot(req.dataset_path)
        if req.dataset_path and Path(req.dataset_path).exists()
        else generate_synthetic_dataset(seed=req.seed)
    )
    try:
        policy = build_policy(req.policy_id, ppo_model_path=settings.rl.model_output_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registry = ExperimentRegistry(settings.duckdb_path)
    run = run_single_episode(
        dataset=dataset,
        scenario=scenarios[req.scenario_id],
        policy=policy,
        seed=req.seed,
        settings=settings,
        experiment_name=req.experiment_name,
        registry=registry,
    )
    return run.model_dump(mode="json")


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Run multi-seed paired backtest across scenarios and policies."""
    settings = load_app_settings()
    scenarios_map = load_scenarios()
    selected_scenarios = []
    for sid in req.scenario_ids:
        if sid not in scenarios_map:
            raise HTTPException(status_code=404, detail=f"Scenario '{sid}' not found")
        selected_scenarios.append(scenarios_map[sid])

    dataset = (
        load_dataset_snapshot(req.dataset_path)
        if req.dataset_path and Path(req.dataset_path).exists()
        else generate_synthetic_dataset(seed=42)
    )

    out = run_backtest_suite(
        dataset=dataset,
        scenarios=selected_scenarios,
        policy_ids=req.policy_ids,
        seeds=req.seeds,
        settings=settings,
        experiment_name=req.experiment_name,
        baseline_policy_id=req.baseline_policy_id,
    )
    summary_df = out["summary_df"]
    comparison_df = out["comparison_df"]
    return BacktestResponse(
        experiment_name=req.experiment_name,
        run_count=len(out["runs"]),
        csv_path=str(out["csv_path"]),
        parquet_path=str(out["parquet_path"]),
        html_path=str(out["html_path"]),
        pdf_path=str(out["pdf_path"]),
        summary=list(summary_df.to_dict(orient="records")),
        comparison=list(comparison_df.to_dict(orient="records")),
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
    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    df = registry.get_run_events(run_id=run_id, limit=limit)
    return list(df.to_dict(orient="records"))


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
    orig_pts = [GeoPoint.model_validate(o) for o in req.origins]
    dest_pts = [GeoPoint.model_validate(d) for d in req.destinations]
    matrix = get_route_matrix(
        origins=orig_pts,
        destinations=dest_pts,
        provider=req.provider,
        profile=req.profile,
        departure_minute=req.departure_minute,
    )
    return matrix.model_dump(mode="json")
