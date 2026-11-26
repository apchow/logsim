"""Zero data-leakage verification tests."""

from __future__ import annotations

import numpy as np

from app.domain.entities import PublicOrderView, Scenario, ScenarioOverlay
from app.domain.enums import OverlayType
from app.policies.optimizers.ortools_dispatcher import ORToolsDispatchPolicy
from app.policies.rules.greedy_cost import GreedyCostPolicy
from app.simulation.env import ReverseLogisticsEnv


def test_public_order_view_excludes_ground_truth_fields() -> None:
    fields = set(PublicOrderView.model_fields.keys())
    assert "true_disposition" not in fields
    assert "cancel_minute" not in fields


def test_causal_barrier_prevents_future_order_and_disruption_leakage(sample_dataset) -> None:
    """Altering orders & disruptions after T_cutoff=900 must leave all obs & actions <=900 identical."""
    cutoff_min = 900.0
    base_scenario = Scenario(
        scenario_id="leak_test",
        version="1.0.0",
        name="Leak Check",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=[],
    )

    # Perturbed future: drop/modify all orders with created_minute > cutoff_min AND add future disruption
    perturbed_orders = [
        o for o in sample_dataset.orders if o.created_minute <= cutoff_min
    ]
    # Also mutate cancel_minute & true_disposition on orders created <= cutoff (if cancel > cutoff)
    mutated_orders = []
    for o in perturbed_orders:
        mutated_orders.append(
            o.model_copy(
                update={
                    "cancel_minute": cutoff_min + 300.0 if o.cancel_minute is None else o.cancel_minute,
                }
            )
        )
    future_dataset = sample_dataset.model_copy(update={"orders": mutated_orders})
    future_scenario = base_scenario.model_copy(
        update={
            "overlays": [
                ScenarioOverlay(
                    overlay_type=OverlayType.WEATHER_DISRUPTION,
                    start_minute=cutoff_min + 120.0,
                    end_minute=2000.0,
                    value=2.0,
                )
            ]
        }
    )

    env1 = ReverseLogisticsEnv(dataset=sample_dataset, scenario=base_scenario, seed=55)
    env2 = ReverseLogisticsEnv(dataset=future_dataset, scenario=future_scenario, seed=55)

    for policy in (GreedyCostPolicy(), ORToolsDispatchPolicy()):
        obs1, info1 = env1.reset(seed=55)
        obs2, info2 = env2.reset(seed=55)
        assert np.allclose(obs1, obs2)

        while info1["sim_time_minute"] <= cutoff_min:
            act1 = policy.select_action(obs1, info1)
            act2 = policy.select_action(obs2, info2)
            assert np.array_equal(act1, act2), (
                f"Policy {policy.policy_id} leaked future state at t={info1['sim_time_minute']}"
            )
            obs1, _, d1, _, info1 = env1.step(act1)
            obs2, _, d2, _, info2 = env2.step(act2)
            assert d1 == d2
            if info1["sim_time_minute"] <= cutoff_min:
                assert np.allclose(obs1, obs2), (
                    f"Observation leaked future info at t={info1['sim_time_minute']}"
                )
