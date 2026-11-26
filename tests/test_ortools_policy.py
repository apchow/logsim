"""Tests for Google OR-Tools CP-SAT deterministic optimization baseline."""

from __future__ import annotations

from app.experiments.runner import run_single_episode
from app.policies.optimizers.ortools_dispatcher import ORToolsDispatchPolicy


def test_ortools_policy_beats_current_replay(sample_dataset, baseline_scenario, test_settings) -> None:
    from app.policies.rules.replay import CurrentPolicyReplay

    replay_run = run_single_episode(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        policy=CurrentPolicyReplay(),
        seed=202,
        settings=test_settings,
        experiment_name="test_ortools",
    )
    opt_run = run_single_episode(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        policy=ORToolsDispatchPolicy(),
        seed=202,
        settings=test_settings,
        experiment_name="test_ortools",
    )

    assert opt_run.metrics.constraint_violations == 0
    assert opt_run.metrics.on_time_delivery_rate >= replay_run.metrics.on_time_delivery_rate
    assert opt_run.metrics.completed_orders >= replay_run.metrics.completed_orders
