"""Tests for multi-seed backtest, paired statistics, and CSV/Parquet/HTML/PDF reporting."""

from __future__ import annotations

from pathlib import Path

from app.domain.entities import Scenario, ScenarioOverlay
from app.domain.enums import OverlayType
from app.experiments.runner import run_backtest_suite


def test_multi_seed_backtest_and_report_exports(sample_dataset, baseline_scenario, test_settings) -> None:
    surge = Scenario(
        scenario_id="test_surge",
        version="1.0.0",
        name="Surge",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=[ScenarioOverlay(overlay_type=OverlayType.DEMAND_MULTIPLIER, start_minute=0.0, value=1.3)],
    )
    storm = Scenario(
        scenario_id="test_storm",
        version="1.0.0",
        name="Storm",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=[ScenarioOverlay(overlay_type=OverlayType.WEATHER_DISRUPTION, start_minute=300.0, end_minute=1200.0, value=1.5)],
    )

    out = run_backtest_suite(
        dataset=sample_dataset,
        scenarios=[baseline_scenario, surge, storm],
        policy_ids=["current_replay", "fifo_edf", "ortools_optimizer"],
        seeds=[101, 202, 303, 404, 505],
        settings=test_settings,
        experiment_name="test_suite",
        baseline_policy_id="current_replay",
    )

    assert len(out["runs"]) == 45  # 3 scenarios x 3 policies x 5 seeds
    assert not out["summary_df"].empty
    assert not out["comparison_df"].empty
    assert Path(out["csv_path"]).exists()
    assert Path(out["parquet_path"]).exists()
    assert Path(out["html_path"]).exists()
    assert Path(out["pdf_path"]).exists()
    assert Path(out["pdf_path"]).stat().st_size > 1000
