"""Reproducible DuckDB + Parquet experiment registry."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from app.domain.entities import ExperimentRun
from app.simulation.event_log import EventLog


def resolve_git_commit() -> str:
    """Return current HEAD commit SHA or fallback if repo has no commits yet."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "uncommitted"


def resolve_git_dirty() -> bool:
    """Return True if working tree has uncommitted changes."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0:
            return bool(res.stdout.strip())
    except Exception:
        pass
    return False


def resolve_lock_hash() -> str | None:
    """Return SHA-256 prefix of uv.lock if present for environment reproducibility."""
    for candidate in (Path("uv.lock"), Path(__file__).resolve().parents[2] / "uv.lock"):
        if candidate.exists():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]
    return None


class ExperimentRegistry:
    """Stores reproducible experiment metadata, KPI results, and immutable event logs in DuckDB."""

    def __init__(self, db_path: str | Path = "artifacts/registry.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id              VARCHAR PRIMARY KEY,
                    experiment_name     VARCHAR NOT NULL,
                    config_hash         VARCHAR NOT NULL,
                    git_commit          VARCHAR NOT NULL,
                    git_dirty           BOOLEAN NOT NULL DEFAULT FALSE,
                    dependency_lock_hash VARCHAR,
                    data_snapshot_id    VARCHAR NOT NULL,
                    seed                INTEGER NOT NULL,
                    policy_id           VARCHAR NOT NULL,
                    policy_type         VARCHAR NOT NULL,
                    policy_version      VARCHAR NOT NULL,
                    scenario_id         VARCHAR NOT NULL,
                    scenario_version    VARCHAR NOT NULL,
                    status              VARCHAR NOT NULL,
                    started_at          TIMESTAMP NOT NULL,
                    finished_at         TIMESTAMP NOT NULL,
                    runtime_seconds     DOUBLE NOT NULL,
                    metrics_json        JSON NOT NULL,
                    artifact_paths_json JSON NOT NULL,
                    mlflow_run_id       VARCHAR
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kpi_results (
                    run_id                    VARCHAR PRIMARY KEY,
                    experiment_name           VARCHAR NOT NULL,
                    policy_id                 VARCHAR NOT NULL,
                    policy_type               VARCHAR NOT NULL,
                    policy_version            VARCHAR NOT NULL,
                    scenario_id               VARCHAR NOT NULL,
                    scenario_version          VARCHAR NOT NULL,
                    seed                      INTEGER NOT NULL,
                    config_hash               VARCHAR NOT NULL,
                    git_commit                VARCHAR NOT NULL,
                    data_snapshot_id          VARCHAR NOT NULL,
                    runtime_seconds           DOUBLE NOT NULL,
                    total_orders              INTEGER NOT NULL,
                    completed_orders          INTEGER NOT NULL,
                    on_time_orders            INTEGER NOT NULL,
                    late_orders               INTEGER NOT NULL,
                    cancelled_orders          INTEGER NOT NULL,
                    on_time_delivery_rate     DOUBLE NOT NULL,
                    fill_rate                 DOUBLE NOT NULL,
                    cost_per_order            DOUBLE NOT NULL,
                    total_cost                DOUBLE NOT NULL,
                    net_cost_after_recovery   DOUBLE NOT NULL,
                    recovery_value            DOUBLE NOT NULL,
                    resource_utilization      DOUBLE NOT NULL,
                    dock_utilization          DOUBLE NOT NULL,
                    inspector_utilization     DOUBLE NOT NULL,
                    throughput_orders_per_day DOUBLE NOT NULL,
                    avg_cycle_time_minutes    DOUBLE NOT NULL,
                    p95_cycle_time_minutes    DOUBLE NOT NULL,
                    stockouts                 INTEGER NOT NULL,
                    backorders                INTEGER NOT NULL,
                    empty_miles_km            DOUBLE NOT NULL,
                    total_miles_km            DOUBLE NOT NULL,
                    carbon_kg_co2             DOUBLE NOT NULL,
                    constraint_violations     INTEGER NOT NULL,
                    exceptions_raised         INTEGER NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_log (
                    event_id          VARCHAR NOT NULL,
                    sequence_no       BIGINT NOT NULL,
                    timestamp_minute  DOUBLE NOT NULL,
                    event_type        VARCHAR NOT NULL,
                    entity_id         VARCHAR NOT NULL,
                    entity_type       VARCHAR NOT NULL,
                    location_id       VARCHAR,
                    order_id          VARCHAR,
                    resource_id       VARCHAR,
                    run_id            VARCHAR NOT NULL,
                    scenario_id       VARCHAR NOT NULL,
                    seed              INTEGER NOT NULL,
                    payload_json      JSON NOT NULL,
                    PRIMARY KEY (run_id, sequence_no)
                );
                """
            )

    def record_run(self, run: ExperimentRun, event_log: EventLog | None = None) -> None:
        """Persist an ExperimentRun, its KPI row, and optionally its event stream."""
        m = run.metrics
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_runs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    run.run_id,
                    run.experiment_name,
                    run.config_hash,
                    run.git_commit,
                    run.git_dirty,
                    run.dependency_lock_hash,
                    run.data_snapshot_id,
                    run.seed,
                    run.policy_id,
                    run.policy_type,
                    run.policy_version,
                    run.scenario_id,
                    run.scenario_version,
                    run.status,
                    run.started_at.replace(tzinfo=None),
                    run.finished_at.replace(tzinfo=None),
                    run.runtime_seconds,
                    m.model_dump_json(),
                    json.dumps(run.artifact_paths),
                    run.mlflow_run_id,
                ],
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO kpi_results VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    run.run_id,
                    run.experiment_name,
                    run.policy_id,
                    run.policy_type,
                    run.policy_version,
                    run.scenario_id,
                    run.scenario_version,
                    run.seed,
                    run.config_hash,
                    run.git_commit,
                    run.data_snapshot_id,
                    run.runtime_seconds,
                    m.total_orders,
                    m.completed_orders,
                    m.on_time_orders,
                    m.late_orders,
                    m.cancelled_orders,
                    m.on_time_delivery_rate,
                    m.fill_rate,
                    m.cost_per_order,
                    m.total_cost,
                    m.net_cost_after_recovery,
                    m.recovery_value,
                    m.resource_utilization,
                    m.dock_utilization,
                    m.inspector_utilization,
                    m.throughput_orders_per_day,
                    m.avg_cycle_time_minutes,
                    m.p95_cycle_time_minutes,
                    m.stockouts,
                    m.backorders,
                    m.empty_miles_km,
                    m.total_miles_km,
                    m.carbon_kg_co2,
                    m.constraint_violations,
                    m.exceptions_raised,
                ],
            )
            if event_log is not None and len(event_log) > 0:
                ev_df = event_log.to_polars().to_pandas()
                conn.execute("DELETE FROM event_log WHERE run_id = ?", [run.run_id])
                conn.register("tmp_ev_df", ev_df)
                conn.execute("INSERT INTO event_log SELECT * FROM tmp_ev_df")
                conn.unregister("tmp_ev_df")

    def get_kpi_dataframe(self, experiment_name: str | None = None) -> pd.DataFrame:
        """Fetch KPI rows optionally filtered by experiment_name."""
        with self._connect() as conn:
            if experiment_name:
                return conn.execute(
                    "SELECT * FROM kpi_results WHERE experiment_name = ? ORDER BY scenario_id, policy_id, seed",
                    [experiment_name],
                ).fetchdf()
            return conn.execute(
                "SELECT * FROM kpi_results ORDER BY experiment_name, scenario_id, policy_id, seed"
            ).fetchdf()

    def list_runs(self, experiment_name: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """List stored runs from experiment_runs."""
        with self._connect() as conn:
            if experiment_name:
                df = conn.execute(
                    "SELECT * FROM experiment_runs WHERE experiment_name = ? ORDER BY started_at DESC LIMIT ?",
                    [experiment_name, limit],
                ).fetchdf()
            else:
                df = conn.execute(
                    "SELECT * FROM experiment_runs ORDER BY started_at DESC LIMIT ?",
                    [limit],
                ).fetchdf()
        return list(df.to_dict(orient="records"))

    def get_run_events(self, run_id: str, limit: int = 500) -> pd.DataFrame:
        """Query immutable events for a specific run_id in time order."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM event_log WHERE run_id = ? ORDER BY sequence_no ASC LIMIT ?",
                [run_id, limit],
            ).fetchdf()
