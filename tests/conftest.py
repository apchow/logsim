"""Shared pytest fixtures for Logistics Backtester tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AppSettings, RLSettings, SimulationSettings
from app.data.synthetic import generate_synthetic_dataset
from app.domain.entities import NetworkDataset, Scenario


@pytest.fixture
def sample_dataset() -> NetworkDataset:
    """Compact deterministic 2-day reverse-logistics dataset for fast tests."""
    return generate_synthetic_dataset(
        seed=42,
        horizon_minutes=2880.0,
        orders_per_day=60,
        dataset_id="test_reverse_v1",
    )


@pytest.fixture
def baseline_scenario() -> Scenario:
    return Scenario(
        scenario_id="test_baseline",
        version="1.0.0",
        name="Test Baseline",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=[],
        is_synthetic=True,
    )


@pytest.fixture
def test_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        artifacts_dir=tmp_path / "artifacts",
        data_dir=tmp_path / "data",
        duckdb_path=tmp_path / "artifacts" / "registry.duckdb",
        mlflow_tracking_uri=(tmp_path / "artifacts" / "mlruns").as_uri(),
        mlflow_enabled=False,
        default_seeds=[101, 202, 303, 404, 505],
        simulation=SimulationSettings(horizon_minutes=2880.0, decision_interval_minutes=60.0),
        rl=RLSettings(
            total_timesteps=512,
            n_steps=128,
            batch_size=32,
            n_epochs=2,
            net_arch=[32, 32],
            model_output_path=str(tmp_path / "artifacts" / "models" / "ppo_test.zip"),
        ),
    )
