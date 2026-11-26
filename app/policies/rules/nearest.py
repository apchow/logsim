"""Nearest-available-resource dispatching baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext


class NearestVehiclePolicy(BasePolicy):
    """Assigns each pending return to the nearest feasible vehicle."""

    policy_id = "nearest_resource"
    policy_type = PolicyType.RULE_BASED
    policy_version = "1.0.0"

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        tracker = EpochCapacityTracker(ctx, vehicle_ids)

        for slot_idx, order in enumerate(ctx.visible_orders):
            candidates = [vid for vid in vehicle_ids if tracker.can_assign(vid, order)]
            if not candidates:
                continue
            best_vid = min(
                candidates,
                key=lambda vid: ctx.lane_distances_km.get(
                    (tracker.veh_map[vid].current_location_id, order.origin_id), 999.0
                ),
            )
            action[slot_idx] = tracker.commit(best_vid, order)
        return action
