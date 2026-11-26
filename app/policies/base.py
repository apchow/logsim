"""Base policy interface and shared causal feasibility tracking for dispatch policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.domain.entities import PublicOrderView
from app.domain.enums import HandlingClass, PolicyType
from app.simulation.engine import DecisionContext, VehicleCausalView


class EpochCapacityTracker:
    """Tracks remaining per-vehicle capacity and HOS inside a single decision epoch."""

    def __init__(self, ctx: DecisionContext, vehicle_ids: list[str]) -> None:
        self.ctx = ctx
        self.vehicle_ids = vehicle_ids
        self.veh_index = {vid: idx + 1 for idx, vid in enumerate(vehicle_ids)}
        self.veh_map = {v.vehicle_id: v for v in ctx.vehicles}
        self.rem_kg: dict[str, float] = {v.vehicle_id: v.capacity_kg for v in ctx.vehicles}
        self.rem_m3: dict[str, float] = {v.vehicle_id: v.capacity_m3 for v in ctx.vehicles}
        self.rem_units: dict[str, int] = {v.vehicle_id: v.capacity_units for v in ctx.vehicles}
        self.rem_hos: dict[str, float] = {
            v.vehicle_id: v.remaining_shift_minutes for v in ctx.vehicles
        }
        self.assigned_counts: dict[str, int] = {v.vehicle_id: 0 for v in ctx.vehicles}

    def estimate_leg_minutes(self, vehicle: VehicleCausalView, order: PublicOrderView) -> float:
        """Estimate incremental route duration using strictly causal lane base minutes."""
        d1 = self.ctx.lane_base_minutes.get((vehicle.current_location_id, order.origin_id), 20.0)
        d2 = self.ctx.lane_base_minutes.get((order.origin_id, order.destination_hub_id), 20.0)
        base = d1 + d2 + 15.0
        return base if self.assigned_counts[vehicle.vehicle_id] == 0 else base * 0.45

    def can_assign(self, vid: str, order: PublicOrderView) -> bool:
        """Return True iff assigning order to vehicle vid satisfies all hard constraints."""
        v = self.veh_map.get(vid)
        if v is None or not v.available or v.broken_down:
            return False
        if (
            order.requires_safety_segregation
            or order.handling_class == HandlingClass.HAZMAT_BATTERY
        ) and not v.hazmat_certified:
            return False
        if order.total_weight_kg > self.rem_kg[vid] + 1e-6:
            return False
        if order.total_volume_m3 > self.rem_m3[vid] + 1e-6:
            return False
        if order.quantity > self.rem_units[vid]:
            return False
        return self.estimate_leg_minutes(v, order) <= self.rem_hos[vid]

    def commit(self, vid: str, order: PublicOrderView) -> int:
        """Reserve capacity for order on vehicle vid and return Gymnasium action code."""
        v = self.veh_map[vid]
        est_min = self.estimate_leg_minutes(v, order)
        self.rem_kg[vid] -= order.total_weight_kg
        self.rem_m3[vid] -= order.total_volume_m3
        self.rem_units[vid] -= order.quantity
        self.rem_hos[vid] -= est_min
        self.assigned_counts[vid] += 1
        return self.veh_index[vid]


class BasePolicy(ABC):
    """Common Gymnasium-compatible interface implemented by every policy."""

    policy_id: str
    policy_type: PolicyType
    policy_version: str = "1.0.0"

    def reset(self, seed: int | None = None) -> None:
        """Reset optional internal policy state between episodes."""
        _ = seed

    @abstractmethod
    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        """Choose an action matching ReverseLogisticsEnv.action_space."""

    @staticmethod
    def empty_action(info: dict[str, Any], default_len: int = 10) -> np.ndarray:
        """Return an all-defer MultiDiscrete action array."""
        ctx: DecisionContext | None = info.get("context")
        n = max(default_len, len(ctx.visible_orders)) if ctx is not None else default_len
        return np.zeros(n, dtype=np.int64)
