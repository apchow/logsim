"""Application configuration, YAML loading, and deterministic config hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SimulationSettings(BaseModel):
    """Configuration for the discrete-event reverse-logistics engine."""

    horizon_minutes: float = Field(default=4320.0, gt=0)  # 3 days default fast demo, 7 days configurable
    decision_interval_minutes: float = Field(default=60.0, gt=0)
    max_visible_orders: int = Field(default=10, ge=1, le=50)
    max_vehicles_in_obs: int = Field(default=6, ge=1, le=30)
    overflow_carrier_cost_per_order: float = Field(default=42.0, ge=0)
    overflow_carrier_transit_minutes: float = Field(default=240.0, ge=0)
    late_penalty_per_hour: float = Field(default=8.0, ge=0)
    unserved_penalty_per_order: float = Field(default=55.0, ge=0)
    violation_penalty_per_attempt: float = Field(default=35.0, ge=0)
    holding_cost_per_unit_hour: float = Field(default=0.05, ge=0)
    stockout_penalty_per_unit: float = Field(default=25.0, ge=0)
    empty_mile_cost_factor: float = Field(default=0.45, ge=0)


class RLSettings(BaseModel):
    """Stable-Baselines3 PPO training hyperparameters."""

    total_timesteps: int = Field(default=25_000, ge=128)
    learning_rate: float = Field(default=3e-4, gt=0)
    n_steps: int = Field(default=512, ge=32)
    batch_size: int = Field(default=64, ge=8)
    n_epochs: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.99, gt=0, le=1.0)
    gae_lambda: float = Field(default=0.95, gt=0, le=1.0)
    clip_range: float = Field(default=0.2, gt=0, le=1.0)
    ent_coef: float = Field(default=0.01, ge=0)
    net_arch: list[int] = Field(default_factory=lambda: [128, 128])
    model_output_path: str = "artifacts/models/ppo_reverse_dispatcher.zip"


class AppSettings(BaseSettings):
    """Top-level platform settings."""

    model_config = SettingsConfigDict(env_prefix="LOGISTICSBT_", extra="ignore")

    project_name: str = "logistics-backtester"
    domain: str = "reverse_pickup_and_hub_triage"
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    duckdb_path: Path = Path("artifacts/registry.duckdb")
    mlflow_tracking_uri: str = "file:./artifacts/mlruns"
    mlflow_enabled: bool = True
    default_seeds: list[int] = Field(default_factory=lambda: [101, 202, 303, 404, 505])
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    rl: RLSettings = Field(default_factory=RLSettings)

    def ensure_dirs(self) -> None:
        """Create required artifact and data directories."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "events").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "results").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "models").mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "synthetic").mkdir(parents=True, exist_ok=True)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file safely."""
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at root of YAML config {path}")
    return loaded


def load_app_settings(config_path: str | Path | None = None) -> AppSettings:
    """Build AppSettings optionally merged with a YAML file."""
    if config_path is None:
        default_cfg = Path("configs/base.yaml")
        if not default_cfg.exists():
            default_cfg = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"
        if default_cfg.exists():
            config_path = default_cfg
    if config_path is not None and Path(config_path).exists():
        raw = load_yaml_config(config_path)
        return AppSettings(**raw)
    return AppSettings()


def compute_config_hash(payload: BaseModel | dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of a configuration dictionary or model."""
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
