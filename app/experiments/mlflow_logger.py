"""MLflow local/remote experiment tracking logger."""

from __future__ import annotations

from pathlib import Path

from app.domain.entities import ExperimentRun


def _normalize_tracking_uri(uri: str) -> str:
    if uri.startswith("file:"):
        raw_path = uri.removeprefix("file:")
        if raw_path.startswith("//"):
            return uri
        return Path(raw_path).resolve().as_uri()
    return uri


def log_run_to_mlflow(run: ExperimentRun, tracking_uri: str = "file:./artifacts/mlruns") -> str | None:
    """Log an ExperimentRun to MLflow and return the MLflow run_id if successful."""
    try:
        import mlflow

        resolved_uri = _normalize_tracking_uri(tracking_uri)
        mlflow.set_tracking_uri(resolved_uri)
        mlflow.set_experiment(run.experiment_name)
        with mlflow.start_run(run_name=f"{run.policy_id}__{run.scenario_id}__s{run.seed}") as active:
            mlflow.log_params(
                {
                    "run_id": run.run_id,
                    "config_hash": run.config_hash,
                    "git_commit": run.git_commit,
                    "data_snapshot_id": run.data_snapshot_id,
                    "seed": run.seed,
                    "policy_id": run.policy_id,
                    "policy_type": run.policy_type,
                    "policy_version": run.policy_version,
                    "scenario_id": run.scenario_id,
                    "scenario_version": run.scenario_version,
                }
            )
            mlflow.log_metrics(
                {
                    "on_time_delivery_rate": run.metrics.on_time_delivery_rate,
                    "fill_rate": run.metrics.fill_rate,
                    "total_cost": run.metrics.total_cost,
                    "cost_per_order": run.metrics.cost_per_order,
                    "net_cost_after_recovery": run.metrics.net_cost_after_recovery,
                    "resource_utilization": run.metrics.resource_utilization,
                    "empty_miles_km": run.metrics.empty_miles_km,
                    "carbon_kg_co2": run.metrics.carbon_kg_co2,
                    "constraint_violations": float(run.metrics.constraint_violations),
                    "runtime_seconds": run.runtime_seconds,
                }
            )
            return str(active.info.run_id)
    except Exception:
        return None
