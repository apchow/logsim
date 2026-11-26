"""Gymnasium-compatible environment wrapping the SimPy reverse-logistics simulator."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.config import SimulationSettings
from app.domain.entities import (
    NetworkDataset,
    OrderAssignment,
    PolicyDecision,
    Scenario,
)
from app.domain.enums import HandlingClass
from app.simulation.engine import DecisionContext, ReverseLogisticsSimulator, StepMetricsDelta


class ReverseLogisticsEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment exposing reset(), step(action), observation_space, action_space."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        dataset: NetworkDataset,
        scenario: Scenario,
        sim_settings: SimulationSettings | None = None,
        policy_id: str = "gym_agent",
        policy_type: str = "rl",
        policy_version: str = "1.0.0",
        seed: int = 42,
        run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.base_dataset = dataset
        self.scenario = scenario
        self.sim_settings = sim_settings or SimulationSettings(
            horizon_minutes=scenario.horizon_minutes,
            decision_interval_minutes=scenario.decision_interval_minutes,
        )
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.policy_version = policy_version
        self._base_seed = seed
        self._run_id = run_id

        self.max_orders = self.sim_settings.max_visible_orders
        self.max_vehicles = min(
            self.sim_settings.max_vehicles_in_obs, max(1, len(dataset.vehicles))
        )
        self.max_hubs = 2

        # 6 global + 10*K orders + 8*M vehicles + 4*H hubs
        self.obs_dim = 6 + 10 * self.max_orders + 8 * self.max_vehicles + 4 * self.max_hubs
        self.observation_space = spaces.Box(
            low=-1.0,
            high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        # Per visible order slot: 0=defer, 1..M=vehicle_m, M+1=overflow_carrier
        self.action_space = spaces.MultiDiscrete([self.max_vehicles + 2] * self.max_orders)

        self.simulator: ReverseLogisticsSimulator | None = None
        self.current_context: DecisionContext | None = None
        self.vehicle_ids: list[str] = sorted(v.vehicle_id for v in dataset.vehicles)[
            : self.max_vehicles
        ]
        self.last_solver_time_ms: float = 0.0

    def set_policy_metadata(
        self, policy_id: str, policy_type: str, policy_version: str = "1.0.0"
    ) -> None:
        """Update active policy metadata attached to PolicyDecision logs."""
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.policy_version = policy_version

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset SimPy simulation deterministically and return initial leak-free observation."""
        super().reset(seed=seed)
        resolved_seed = seed if seed is not None else self._base_seed
        self._base_seed = resolved_seed

        scenario = self.scenario
        if options and isinstance(options.get("scenario"), Scenario):
            scenario = options["scenario"]
            self.scenario = scenario

        self.simulator = ReverseLogisticsSimulator(
            dataset=self.base_dataset,
            scenario=scenario,
            seed=resolved_seed,
            sim_settings=self.sim_settings,
            run_id=self._run_id,
        )
        self.vehicle_ids = sorted(self.simulator.vehicles.keys())[: self.max_vehicles]
        self.current_context = self.simulator.advance_to_first_decision()
        obs = self._encode_observation(self.current_context)
        info: dict[str, Any] = {
            "context": self.current_context,
            "vehicle_ids": list(self.vehicle_ids),
            "sim_time_minute": self.current_context.timestamp_minute,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply policy action through ConstraintGuard, advance SimPy one epoch, return reward."""
        if self.simulator is None or self.current_context is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.shape[0] != self.max_orders:
            raise ValueError(f"Expected action shape ({self.max_orders},), got {arr.shape}")

        decision = self._decode_action(arr, self.current_context)
        next_ctx, delta, terminated = self.simulator.apply_decision_and_step(decision)
        self.current_context = next_ctx

        reward = self._compute_reward(delta, arr, len(decision.assignments))
        obs = self._encode_observation(next_ctx)
        info: dict[str, Any] = {
            "context": next_ctx,
            "vehicle_ids": list(self.vehicle_ids),
            "sim_time_minute": next_ctx.timestamp_minute,
            "step_delta": delta,
            "constraint_violations": delta.constraint_violations,
        }
        return obs, float(reward), terminated, False, info

    def _decode_action(self, action: np.ndarray, ctx: DecisionContext) -> PolicyDecision:
        """Translate MultiDiscrete vector into OrderAssignments without altering illegal picks."""
        assignments: list[OrderAssignment] = []
        n_vis = len(ctx.visible_orders)
        overflow_code = self.max_vehicles + 1

        for idx in range(self.max_orders):
            act = int(action[idx])
            if idx >= n_vis:
                continue
            order = ctx.visible_orders[idx]
            if act <= 0:
                assignments.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
            elif act == overflow_code:
                assignments.append(
                    OrderAssignment(order_id=order.order_id, action_type="overflow_carrier")
                )
            elif 1 <= act <= len(self.vehicle_ids):
                vid = self.vehicle_ids[act - 1]
                assignments.append(
                    OrderAssignment(
                        order_id=order.order_id,
                        vehicle_id=vid,
                        action_type="dispatch",
                    )
                )
            else:
                assignments.append(OrderAssignment(order_id=order.order_id, action_type="defer"))

        ctx_payload = json.dumps(
            {
                "t": ctx.timestamp_minute,
                "epoch": ctx.epoch_index,
                "orders": [o.model_dump(mode="json") for o in ctx.visible_orders],
                "vehicles": [
                    (v.vehicle_id, v.current_location_id, v.available, v.remaining_shift_minutes)
                    for v in ctx.vehicles
                ],
                "disruptions": list(ctx.active_disruptions),
            },
            sort_keys=True,
        )
        obs_hash = hashlib.sha256(ctx_payload.encode("utf-8")).hexdigest()

        return PolicyDecision(
            timestamp_minute=ctx.timestamp_minute,
            epoch_index=ctx.epoch_index,
            policy_id=self.policy_id,
            policy_type=self.policy_type,
            policy_version=self.policy_version,
            observation_hash=obs_hash,
            assignments=assignments,
            solver_time_ms=self.last_solver_time_ms,
        )

    def _encode_observation(self, ctx: DecisionContext) -> np.ndarray:
        """Build fixed-size normalized float32 vector strictly from causal DecisionContext."""
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        hour = (ctx.timestamp_minute % 1440.0) / 60.0
        angle = 2.0 * math.pi * (hour / 24.0)

        mean_dock_util = (
            float(
                np.mean(
                    [h.docks_in_use / max(1, h.total_docks) for h in ctx.hubs]
                )
            )
            if ctx.hubs
            else 0.0
        )

        # Global features [0..5]
        vec[0] = float(np.clip(ctx.timestamp_minute / max(1.0, ctx.horizon_minutes), 0.0, 1.0))
        vec[1] = float(math.sin(angle))
        vec[2] = float(math.cos(angle))
        vec[3] = float(np.clip(ctx.total_pending_backlog / max(1, self.max_orders), 0.0, 10.0))
        vec[4] = float(np.clip(len(ctx.active_disruptions), 0.0, 10.0))
        vec[5] = float(np.clip(mean_dock_util, 0.0, 1.0))

        veh_map = {v.vehicle_id: v for v in ctx.vehicles}
        avail_vehs = [v for v in ctx.vehicles if v.available]

        # Order slots
        offset = 6
        for i in range(self.max_orders):
            if i < len(ctx.visible_orders):
                o = ctx.visible_orders[i]
                time_to_due_hr = (o.promise_window_end_minute - ctx.timestamp_minute) / 60.0
                age_hr = (ctx.timestamp_minute - o.created_minute) / 60.0
                if avail_vehs:
                    nearest_dist = min(
                        ctx.lane_distances_km.get((v.current_location_id, o.origin_id), 25.0)
                        for v in avail_vehs
                    )
                else:
                    nearest_dist = 100.0
                vec[offset + 0] = 1.0
                vec[offset + 1] = float(np.clip(time_to_due_hr / 12.0, -1.0, 5.0))
                vec[offset + 2] = float(np.clip(o.priority / 5.0, 0.0, 1.0))
                vec[offset + 3] = float(np.clip(o.total_weight_kg / 50.0, 0.0, 5.0))
                vec[offset + 4] = float(np.clip(o.total_volume_m3 / 1.0, 0.0, 5.0))
                vec[offset + 5] = 1.0 if o.handling_class == HandlingClass.HAZMAT_BATTERY else 0.0
                vec[offset + 6] = float(np.clip(o.estimated_distance_km / 50.0, 0.0, 5.0))
                vec[offset + 7] = float(np.clip(o.unit_value_usd / 350.0, 0.0, 5.0))
                vec[offset + 8] = float(np.clip(nearest_dist / 50.0, 0.0, 5.0))
                vec[offset + 9] = float(np.clip(age_hr / 12.0, 0.0, 5.0))
            offset += 10

        # Vehicle slots
        for j in range(self.max_vehicles):
            if j < len(self.vehicle_ids):
                v = veh_map[self.vehicle_ids[j]]
                wait_hr = max(0.0, v.busy_until_minute - ctx.timestamp_minute) / 60.0
                vec[offset + 0] = 1.0
                vec[offset + 1] = 1.0 if v.available else 0.0
                vec[offset + 2] = 1.0 if v.broken_down else 0.0
                vec[offset + 3] = 1.0 if v.hazmat_certified else 0.0
                vec[offset + 4] = float(np.clip(v.remaining_shift_minutes / 540.0, 0.0, 1.0))
                vec[offset + 5] = float(np.clip(v.capacity_kg / 250.0, 0.0, 2.0))
                vec[offset + 6] = float(np.clip(wait_hr, 0.0, 5.0))
                vec[offset + 7] = 1.0 if v.current_location_id == v.home_hub_id else 0.0
            offset += 8

        # Hub slots
        for h_idx in range(self.max_hubs):
            if h_idx < len(ctx.hubs):
                h = ctx.hubs[h_idx]
                vec[offset + 0] = float(np.clip(h.docks_in_use / max(1, h.total_docks), 0.0, 1.0))
                vec[offset + 1] = float(
                    np.clip(h.inspection_in_use / max(1, h.total_inspection_bays), 0.0, 1.0)
                )
                vec[offset + 2] = float(np.clip(h.quarantine_units / 100.0, 0.0, 5.0))
                vec[offset + 3] = float(np.clip(h.refurb_kit_units / 100.0, 0.0, 5.0))
            offset += 4

        return np.clip(vec, -1.0, 10.0)

    def _compute_reward(
        self,
        delta: StepMetricsDelta,
        action: np.ndarray,
        n_assigned_orders: int,
    ) -> float:
        """Multi-objective RL reward combining service, cost, utilization, lateness, empty miles, and violations."""
        active_dispatches = int(np.sum((action > 0) & (action <= self.max_vehicles)))
        valid_dispatches = max(0, active_dispatches - delta.constraint_violations)
        # Immediate credit for valid fleet utilization without exceeding constraints
        batch_bonus = 0.35 * min(valid_dispatches, max(0, n_assigned_orders))

        reward = (
            2.2 * delta.on_time_orders
            + 0.015 * delta.recovery_value
            + batch_bonus
            - 0.015 * (delta.transport_cost + delta.processing_cost + delta.holding_cost)
            - 0.04 * delta.empty_miles_km
            - 0.6 * delta.late_orders
            - 0.012 * delta.late_minutes
            - 0.02 * delta.penalty_cost
            - 3.5 * delta.constraint_violations
        )
        return float(reward)

    def render(self) -> str:
        """Return compact human-readable textual summary of current state."""
        if self.current_context is None:
            return "ReverseLogisticsEnv(uninitialized)"
        c = self.current_context
        return (
            f"t={c.timestamp_minute:.0f}m backlog={c.total_pending_backlog} "
            f"avail_veh={sum(1 for v in c.vehicles if v.available)}/{len(c.vehicles)} "
            f"disruptions={len(c.active_disruptions)}"
        )
