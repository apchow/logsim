"""Fixed dispatch batch baseline waiting for consolidation or SLA urgency trigger."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext


class FixedBatchDispatchPolicy(BasePolicy):
    """Holds returns until hub batch size >= threshold or an order nears SLA deadline."""

    policy_id = "fixed_batch"
    policy_type = PolicyType.RULE_BASED
    policy_version = "1.0.0"

    def __init__(self, batch_size: int = 3, urgency_minutes: float = 210.0) -> None:
        self.batch_size = batch_size
        self.urgency_minutes = urgency_minutes

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        tracker = EpochCapacityTracker(ctx, vehicle_ids)

        # Group visible orders by destination hub
        by_hub: dict[str, list[tuple[int, Any]]] = {}
        for slot_idx, order in enumerate(ctx.visible_orders):
            by_hub.setdefault(order.destination_hub_id, []).append((slot_idx, order))

        for hub_id, items in by_hub.items():
            urgent = any(
                (o.promise_window_end_minute - ctx.timestamp_minute) <= self.urgency_minutes
                for _, o in items
            )
            if len(items) < self.batch_size and not urgent:
                continue
            candidates = sorted(
                vehicle_ids,
                key=lambda vid: (0 if tracker.veh_map[vid].home_hub_id == hub_id else 1, vid),
            )
            for slot_idx, order in items:
                for vid in candidates:
                    if tracker.can_assign(vid, order):
                        action[slot_idx] = tracker.commit(vid, order)
                        break
        return action
