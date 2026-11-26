"""Tests for rule-based, replay, do-nothing, and ML risk-scorer policies."""

from __future__ import annotations

import pytest

from app.experiments.runner import run_single_episode


@pytest.mark.parametrize(
    "policy_id",
    [
        "do_nothing",
        "current_replay",
        "fifo_edf",
        "nearest_resource",
        "fixed_batch",
        "greedy_cost",
        "ml_risk_scorer",
    ],
)
def test_rule_and_ml_policies_execute_without_violations(
    policy_id: str,
    sample_dataset,
    baseline_scenario,
    test_settings,
) -> None:
    from app.policies.registry import build_policy

    policy = build_policy(policy_id)
    run = run_single_episode(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        policy=policy,
        seed=101,
        settings=test_settings,
        experiment_name="test_policies",
    )
    assert run.status == "completed"
    assert run.metrics.constraint_violations == 0
    if policy_id != "do_nothing":
        assert run.metrics.completed_orders > 0
        assert run.metrics.on_time_delivery_rate > 0.0
    else:
        assert run.metrics.completed_orders == 0
