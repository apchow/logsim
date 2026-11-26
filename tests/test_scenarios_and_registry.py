"""Tests for all 11 scenario overlays and DuckDB/Parquet experiment registry."""

from __future__ import annotations

from app.domain.entities import Scenario, ScenarioOverlay
from app.domain.enums import OverlayType
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import run_single_episode
from app.policies.rules.fifo_edf import FIFODispatchPolicy


def test_all_eleven_scenario_overlays_supported(sample_dataset, test_settings) -> None:
    overlays = [
        ScenarioOverlay(overlay_type=OverlayType.DEMAND_MULTIPLIER, start_minute=0.0, value=1.2),
        ScenarioOverlay(overlay_type=OverlayType.DEMAND_MIX_SHIFT, start_minute=0.0, value=0.3),
        ScenarioOverlay(overlay_type=OverlayType.CARRIER_DELAY, start_minute=200.0, end_minute=600.0, value=1.3),
        ScenarioOverlay(overlay_type=OverlayType.LEAD_TIME_INCREASE, start_minute=200.0, end_minute=600.0, value=1.2),
        ScenarioOverlay(overlay_type=OverlayType.WAREHOUSE_CAPACITY_REDUCTION, start_minute=300.0, end_minute=700.0, target_id="HUB-NORTH", value=0.4),
        ScenarioOverlay(overlay_type=OverlayType.LABOR_ABSENCE, start_minute=300.0, end_minute=700.0, value=0.3),
        ScenarioOverlay(overlay_type=OverlayType.VEHICLE_BREAKDOWN, start_minute=400.0, end_minute=800.0, target_id="VEH-01", value=1.0),
        ScenarioOverlay(overlay_type=OverlayType.PORT_CUSTOMS_DELAY, start_minute=200.0, end_minute=900.0, value=1.5),
        ScenarioOverlay(overlay_type=OverlayType.WEATHER_DISRUPTION, start_minute=500.0, end_minute=900.0, value=1.4),
        ScenarioOverlay(overlay_type=OverlayType.SKU_AVAILABILITY_SHOCK, start_minute=600.0, end_minute=1200.0, value=0.8),
        ScenarioOverlay(overlay_type=OverlayType.NEW_CUSTOMER_PROMISE_POLICY, start_minute=0.0, value=0.85),
    ]
    mega_scenario = Scenario(
        scenario_id="all_11_overlays",
        version="1.0.0",
        name="All 11 Overlays Stress Test",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=overlays,
    )
    reg = ExperimentRegistry(test_settings.duckdb_path)
    run = run_single_episode(
        dataset=sample_dataset,
        scenario=mega_scenario,
        policy=FIFODispatchPolicy(),
        seed=101,
        settings=test_settings,
        experiment_name="test_registry_exp",
        registry=reg,
    )

    df = reg.get_kpi_dataframe("test_registry_exp")
    assert len(df) == 1
    assert df.iloc[0]["config_hash"] == run.config_hash
    assert df.iloc[0]["data_snapshot_id"] == sample_dataset.snapshot_id
    events_df = reg.get_run_events(run.run_id, limit=200)
    assert not events_df.empty
