"""FIFO / Earliest-Deadline-First (EDF) dispatching baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext


class FIFODispatchPolicy(BasePolicy):
    """Assigns earliest-deadline-first orders to first feasible vehicle (overflow if critical)."""

    policy_id = "fifo_edf"
    policy_type = PolicyType.RULE_BASED
    policy_version = "1.0.0"

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        tracker = EpochCapacityTracker(ctx, vehicle_ids)
        overflow_code = len(vehicle_ids) + 1

        # ctx.visible_orders is already sorted by (promise_window_end_minute, -priority, created_minute)
        for slot_idx, order in enumerate(ctx.visible_orders):
            assigned = False
            # Prefer home hub vehicles first, then any available vehicle
            ordered_vids = sorted(
                vehicle_ids,
                key=lambda vid: (0 if tracker.veh_map[vid].home_hub_id == order.destination_hub_id else 1, vid),
            )
            for vid in ordered_vids:
                if tracker.can_assign(vid, order):
                    action[slot_idx] = tracker.commit(vid, order)
                    assigned = True
                    break
            if not assigned and (order.promise_window_end_minute - ctx.timestamp_minute) <= 90.0:
                action[slot_idx] = overflow_code
        return action
