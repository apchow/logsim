"""FastAPI request and response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SyntheticDatasetRequest(BaseModel):
    seed: int = 42
    horizon_minutes: float = Field(default=4320.0, gt=0)
    orders_per_day: int = Field(default=110, ge=10)
    dataset_id: str = "synthetic_reverse_v1"
    output_dir: str = "data/synthetic/default"


class SimulateRequest(BaseModel):
    dataset_path: str | None = None
    scenario_id: str = "baseline_normal"
    policy_id: str = "ortools_optimizer"
    seed: int = 101
    experiment_name: str = "api_simulate"


class BacktestRequest(BaseModel):
    dataset_path: str | None = None
    scenario_ids: list[str] = Field(
        default_factory=lambda: ["baseline_normal", "peak_surge_breakdown", "storm_tight_sla"]
    )
    policy_ids: list[str] = Field(
        default_factory=lambda: ["current_replay", "fifo_edf", "greedy_cost", "ortools_optimizer"]
    )
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303, 404, 505])
    experiment_name: str = "api_backtest"
    baseline_policy_id: str = "current_replay"


class BacktestResponse(BaseModel):
    experiment_name: str
    run_count: int
    csv_path: str
    parquet_path: str
    html_path: str
    pdf_path: str
    summary: list[dict[str, Any]]
    comparison: list[dict[str, Any]]


class AnalyzeRequest(BaseModel):
    experiment_name: str
    baseline_policy_id: str = "current_replay"
    question: str | None = None
    model: str = "gemini-4-hydrogen"
