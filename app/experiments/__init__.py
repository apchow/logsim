"""Scenario definitions, multi-seed experiment runner, DuckDB registry, and MLflow logging."""

from app.experiments.registry import (
    ExperimentRegistry,
    resolve_git_commit,
    resolve_git_dirty,
    resolve_lock_hash,
)
from app.experiments.runner import run_backtest_suite, run_single_episode
from app.experiments.scenarios import default_scenarios, load_scenarios

__all__ = [
    "ExperimentRegistry",
    "default_scenarios",
    "load_scenarios",
    "resolve_git_commit",
    "resolve_git_dirty",
    "resolve_lock_hash",
    "run_backtest_suite",
    "run_single_episode",
]
