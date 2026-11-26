"""Causal ML risk & cost scorer policy using scikit-learn."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from app.domain.entities import PublicOrderView
from app.domain.enums import PolicyType
from app.policies.base import BasePolicy, EpochCapacityTracker
from app.simulation.engine import DecisionContext, VehicleCausalView


class MLRiskDispatchPolicy(BasePolicy):
    """Supervised ML scoring policy predicting expected SLA+cost penalty per (order, vehicle)."""

    policy_id = "ml_risk_scorer"
    policy_type = PolicyType.ML
    policy_version = "1.0.0"

    def __init__(self) -> None:
        self.model = Ridge(alpha=1.0)
        self._fit_prior_calibration()

    def _fit_prior_calibration(self) -> None:
        """Fit baseline ridge regressor on synthetic causal feature tuples (zero test-set leakage)."""
        rng = np.random.default_rng(7)
        n = 400
        deadhead = rng.uniform(0.0, 45.0, size=n)
        linehaul = rng.uniform(3.0, 40.0, size=n)
        slack_hr = rng.uniform(-1.0, 14.0, size=n)
        priority = rng.integers(1, 6, size=n).astype(float)
        rem_hos_hr = rng.uniform(0.5, 9.0, size=n)
        disruption_cnt = rng.integers(0, 3, size=n).astype(float)
        same_hub = rng.integers(0, 2, size=n).astype(float)

        X = np.column_stack(
            [deadhead, linehaul, slack_hr, priority, rem_hos_hr, disruption_cnt, same_hub]
        )
        # Cost-plus-lateness target
        y = (
            1.15 * (deadhead + linehaul)
            + 8.0 * np.maximum(0.0, 2.5 - slack_hr) * priority
            + 5.0 * disruption_cnt
            - 4.0 * same_hub
            - 0.8 * rem_hos_hr
            + rng.normal(0.0, 1.5, size=n)
        )
        self.model.fit(X, y)

    @staticmethod
    def _featurize(
        ctx: DecisionContext, order: PublicOrderView, veh: VehicleCausalView
    ) -> np.ndarray:
        deadhead = ctx.lane_distances_km.get((veh.current_location_id, order.origin_id), 25.0)
        linehaul = order.estimated_distance_km
        slack_hr = (order.promise_window_end_minute - ctx.timestamp_minute) / 60.0
        rem_hos_hr = veh.remaining_shift_minutes / 60.0
        disruption_cnt = float(len(ctx.active_disruptions))
        same_hub = 1.0 if veh.home_hub_id == order.destination_hub_id else 0.0
        return np.array(
            [
                deadhead,
                linehaul,
                slack_hr,
                float(order.priority),
                rem_hos_hr,
                disruption_cnt,
                same_hub,
            ],
            dtype=np.float64,
        )

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
                if slack_min <= 90.0:
                    action[slot_idx] = overflow_code
                continue

            feats = np.vstack([self._featurize(ctx, order, tracker.veh_map[vid]) for vid in candidates])
            preds = self.model.predict(feats)
            # Favor batching onto an already-selected vehicle during this epoch
            adjusted = [
                float(preds[idx]) - (8.0 if tracker.assigned_counts[vid] > 0 else 0.0)
                for idx, vid in enumerate(candidates)
            ]
            best_vid = candidates[int(np.argmin(adjusted))]
            if slack_min > 360.0 and min(adjusted) > 48.0 and tracker.assigned_counts[best_vid] == 0:
                continue
            action[slot_idx] = tracker.commit(best_vid, order)
        return action
