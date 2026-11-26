"""Policy factory registry mapping CLI/API policy IDs to instantiated policy objects."""

from __future__ import annotations

from pathlib import Path

from app.policies.base import BasePolicy
from app.policies.ml.risk_scorer import MLRiskDispatchPolicy
from app.policies.optimizers.ortools_dispatcher import ORToolsDispatchPolicy
from app.policies.rl.ppo_policy import RLPPOPolicy
from app.policies.rules.fifo_edf import FIFODispatchPolicy
from app.policies.rules.fixed_batch import FixedBatchDispatchPolicy
from app.policies.rules.greedy_cost import GreedyCostPolicy
from app.policies.rules.nearest import NearestVehiclePolicy
from app.policies.rules.replay import CurrentPolicyReplay, DoNothingPolicy


def available_policy_ids() -> list[str]:
    """Return canonical list of all registered policy IDs."""
    return [
        "do_nothing",
        "current_replay",
        "fifo_edf",
        "nearest_resource",
        "fixed_batch",
        "greedy_cost",
        "ortools_optimizer",
        "ml_risk_scorer",
        "ppo_agent",
    ]


def build_policy(
    policy_id: str,
    ppo_model_path: str | Path = "artifacts/models/ppo_reverse_dispatcher.zip",
) -> BasePolicy:
    """Instantiate a registered policy by ID."""
    mapping: dict[str, BasePolicy] = {
        "do_nothing": DoNothingPolicy(),
        "current_replay": CurrentPolicyReplay(),
        "fifo_edf": FIFODispatchPolicy(),
        "nearest_resource": NearestVehiclePolicy(),
        "fixed_batch": FixedBatchDispatchPolicy(),
        "greedy_cost": GreedyCostPolicy(),
        "ortools_optimizer": ORToolsDispatchPolicy(),
        "ml_risk_scorer": MLRiskDispatchPolicy(),
    }
    if policy_id == "ppo_agent":
        return RLPPOPolicy(model_path=ppo_model_path)
    if policy_id not in mapping:
        raise ValueError(
            f"Unknown policy_id='{policy_id}'. Available: {', '.join(available_policy_ids())}"
        )
    return mapping[policy_id]
