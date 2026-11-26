"""Greedy cost-minimization dispatch baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext


class GreedyCostPolicy(BasePolicy):
    """Minimizes immediate marginal dispatch + distance cost while guarding urgent SLA windows."""

    policy_id = "greedy_cost"
    policy_type = PolicyType.RULE_BASED
    policy_version = "1.0.0"

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        tracker = EpochCapacityTracker(ctx, vehicle_ids)
        overflow_code = len(vehicle_ids) + 1

        for slot_idx, order in enumerate(ctx.visible_orders):
            slack_min = order.promise_window_end_minute - ctx.timestamp_minute
            candidates = [vid for vid in vehicle_ids if tracker.can_assign(vid, order)]
            if not candidates:
                if slack_min <= 120.0:
                    action[slot_idx] = overflow_code
                continue

            def _marginal_score(vid: str, target_origin: str = order.origin_id) -> float:
                v = tracker.veh_map[vid]
                already_dispatched = tracker.assigned_counts[vid] > 0
                deadhead = (
                    0.0
                    if already_dispatched
                    else ctx.lane_distances_km.get((v.current_location_id, target_origin), 25.0)
                )
                fixed_pen = 0.0 if already_dispatched else 9.0
                return fixed_pen + deadhead * v.cost_per_km

            best_vid = min(candidates, key=_marginal_score)
            best_cost = _marginal_score(best_vid)
            # Defer non-urgent orders when deadhead cost is high and no active batch exists
            if slack_min > 300.0 and best_cost > 34.0 and tracker.assigned_counts[best_vid] == 0:
                continue
            action[slot_idx] = tracker.commit(best_vid, order)
        return action
