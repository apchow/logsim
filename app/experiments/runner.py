"""Single-episode and multi-seed paired backtest runners."""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import pandas as pd

from app.config import AppSettings, compute_config_hash
from app.domain.entities import ExperimentRun, NetworkDataset, Scenario
from app.evaluation.kpi import compute_kpis
from app.evaluation.reporting import (
    export_results_csv,
    export_results_parquet,
    generate_html_report,
    generate_pdf_report,
)
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds
from app.experiments.mlflow_logger import log_run_to_mlflow
from app.experiments.registry import (
    ExperimentRegistry,
    resolve_git_commit,
    resolve_git_dirty,
    resolve_lock_hash,
)
from app.policies.base import BasePolicy
from app.policies.registry import build_policy
from app.routing.adapters import RouteMatrix
from app.simulation.env import ReverseLogisticsEnv


def run_single_episode(
    dataset: NetworkDataset,
    scenario: Scenario,
    policy: BasePolicy,
    seed: int,
    settings: AppSettings,
    experiment_name: str = "ad_hoc_sim",
    registry: ExperimentRegistry | None = None,
    persist_events_to_db: bool = True,
    routing_provider: str | None = None,
    routing_profile: str = "driving",
    route_matrix: RouteMatrix | None = None,
) -> ExperimentRun:
    """Run one deterministic episode through the Gymnasium interface and log all artifacts."""
    settings.ensure_dirs()
    run_id = f"{policy.policy_id}__{scenario.scenario_id}__s{seed}__{uuid4().hex[:6]}"
    started_at = datetime.now(UTC)
    t0 = time.perf_counter()

    cache_path = settings.artifacts_dir / "cache" / "route_cache.json" if routing_provider else None
    env = ReverseLogisticsEnv(
        dataset=dataset,
        scenario=scenario,
        sim_settings=settings.simulation,
        policy_id=policy.policy_id,
        policy_type=str(policy.policy_type),
        policy_version=policy.policy_version,
        seed=seed,
        run_id=run_id,
        routing_provider=routing_provider,
        routing_profile=routing_profile,
        route_matrix=route_matrix,
        route_cache_path=cache_path,
    )
    policy.reset(seed=seed)
    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = policy.select_action(obs, info)
        solver_ms = float(getattr(policy, "last_solve_ms", 0.0))
        env.last_solver_time_ms = solver_ms
        obs, _, terminated, truncated, info = env.step(action)

    assert env.simulator is not None
    env.simulator.run_id = run_id
    kpi = compute_kpis(env.simulator, policy_id=policy.policy_id, policy_version=policy.policy_version)
    kpi = kpi.model_copy(update={"run_id": run_id})

    events_path = (
        settings.artifacts_dir
        / "events"
        / f"scenario_id={scenario.scenario_id}"
        / f"policy_id={policy.policy_id}"
        / f"seed={seed}"
        / "events.parquet"
    )
    env.simulator.event_log.to_parquet(events_path)

    runtime_sec = round(time.perf_counter() - t0, 4)
    finished_at = datetime.now(UTC)
    cfg_hash = compute_config_hash(
        {
            "simulation": settings.simulation.model_dump(mode="json"),
            "scenario": scenario.model_dump(mode="json"),
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "seed": seed,
            "snapshot_id": dataset.snapshot_id,
            "routing_provider": env.simulator.routing_provider,
            "route_matrix_sha256": env.simulator.route_matrix_sha256,
        }
    )

    exp_run = ExperimentRun(
        run_id=run_id,
        experiment_name=experiment_name,
        config_hash=cfg_hash,
        git_commit=resolve_git_commit(),
        git_dirty=resolve_git_dirty(),
        dependency_lock_hash=resolve_lock_hash(),
        data_snapshot_id=dataset.snapshot_id,
        seed=seed,
        policy_id=policy.policy_id,
        policy_type=str(policy.policy_type),
        policy_version=policy.policy_version,
        scenario_id=scenario.scenario_id,
        scenario_version=scenario.version,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_sec,
        metrics=kpi,
        artifact_paths={
            "events_parquet": str(events_path),
            "routing_provider": env.simulator.routing_provider,
            "route_matrix_sha256": env.simulator.route_matrix_sha256,
        },
    )

    if settings.mlflow_enabled:
        mlflow_id = log_run_to_mlflow(exp_run, tracking_uri=settings.mlflow_tracking_uri)
        if mlflow_id:
            exp_run = exp_run.model_copy(update={"mlflow_run_id": mlflow_id})

    if registry is not None:
        registry.record_run(exp_run, event_log=env.simulator.event_log if persist_events_to_db else None)

    return exp_run


class BacktestSuiteResult(TypedDict):
    """Typed output bundle returned by run_backtest_suite."""

    experiment_name: str
    runs: list[ExperimentRun]
    results_df: pd.DataFrame
    summary_df: pd.DataFrame
    comparison_df: pd.DataFrame
    csv_path: Path
    parquet_path: Path
    html_path: Path
    pdf_path: Path


def run_backtest_suite(
    dataset: NetworkDataset,
    scenarios: Sequence[Scenario],
    policy_ids: Sequence[str],
    seeds: Sequence[int],
    settings: AppSettings,
    experiment_name: str = "reverse_backtest",
    baseline_policy_id: str = "current_replay",
    routing_provider: str | None = None,
    routing_profile: str = "driving",
) -> BacktestSuiteResult:
    """Run paired multi-seed experiments across policies & scenarios and generate reports."""
    settings.ensure_dirs()
    registry = ExperimentRegistry(settings.duckdb_path)
    runs: list[ExperimentRun] = []

    for scenario in scenarios:
        for policy_id in policy_ids:
            policy = build_policy(policy_id, ppo_model_path=settings.rl.model_output_path)
            for seed in seeds:
                run = run_single_episode(
                    dataset=dataset,
                    scenario=scenario,
                    policy=policy,
                    seed=int(seed),
                    settings=settings,
                    experiment_name=experiment_name,
                    registry=registry,
                    persist_events_to_db=(seed == seeds[0]),
                    routing_provider=routing_provider,
                    routing_profile=routing_profile,
                )
                runs.append(run)

    rows = [
        {
            **r.metrics.model_dump(mode="json"),
            "experiment_name": r.experiment_name,
            "policy_type": r.policy_type,
            "config_hash": r.config_hash,
            "git_commit": r.git_commit,
            "data_snapshot_id": r.data_snapshot_id,
            "runtime_seconds": r.runtime_seconds,
        }
        for r in runs
    ]
    results_df = pd.DataFrame(rows)
    summary_df = summarize_across_seeds(results_df)
    comparison_df = compare_policies_paired(results_df, baseline_policy_id=baseline_policy_id)

    res_dir = settings.artifacts_dir / "results" / experiment_name
    rep_dir = settings.artifacts_dir / "reports" / experiment_name
    csv_path = export_results_csv(results_df, res_dir / "kpi_results.csv")
    pq_path = export_results_parquet(results_df, res_dir / "kpi_results.parquet")
    html_path = generate_html_report(
        results_df, summary_df, comparison_df, rep_dir / "report.html", baseline_policy_id
    )
    pdf_path = generate_pdf_report(
        results_df, summary_df, comparison_df, rep_dir / "report.pdf", baseline_policy_id
    )

    return {
        "experiment_name": experiment_name,
        "runs": runs,
        "results_df": results_df,
        "summary_df": summary_df,
        "comparison_df": comparison_df,
        "csv_path": Path(csv_path),
        "parquet_path": Path(pq_path),
        "html_path": Path(html_path),
        "pdf_path": Path(pdf_path),
    }
