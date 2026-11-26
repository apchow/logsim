"""Current-practice replay (fixed schedule sweep) and pure do-nothing baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext


class DoNothingPolicy(BasePolicy):
    """Pure do-nothing baseline that defers every decision (lower-bound sanity benchmark)."""

    policy_id = "do_nothing"
    policy_type = PolicyType.CURRENT_REPLAY
    policy_version = "1.0.0"

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        return np.zeros(self.empty_action(info).shape, dtype=np.int64)


class CurrentPolicyReplay(BasePolicy):
    """Simulates legacy current practice: static twice-daily hub-bound sweeps in arrival order."""

    policy_id = "current_replay"
    policy_type = PolicyType.CURRENT_REPLAY
    policy_version = "1.0.0"

    def __init__(self, sweep_hours: tuple[int, ...] = (10, 16)) -> None:
        self.sweep_hours = sweep_hours

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        hour = int((ctx.timestamp_minute % 1440.0) // 60.0)
        if hour not in self.sweep_hours:
            return action

        tracker = EpochCapacityTracker(ctx, vehicle_ids)
        # Legacy rule: strictly arrival-ordered and tied to destination home hub vehicles
        indexed = sorted(
            enumerate(ctx.visible_orders),
            key=lambda pair: (pair[1].created_minute, pair[1].order_id),
        )
        for slot_idx, order in indexed:
            for vid in vehicle_ids:
                veh = tracker.veh_map[vid]
                if veh.home_hub_id != order.destination_hub_id:
                    continue
                if tracker.can_assign(vid, order):
                    action[slot_idx] = tracker.commit(vid, order)
                    break
        return action
