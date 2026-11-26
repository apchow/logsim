"""Google OR-Tools CP-SAT deterministic optimization baseline for reverse dispatching."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from ortools.sat.python import cp_model

from app.domain.enums import HandlingClass, PolicyType
from app.policies.base import BasePolicy
from app.simulation.engine import DecisionContext


class ORToolsDispatchPolicy(BasePolicy):
    """Constrained CP-SAT dispatcher minimizing transport, overflow, and SLA deferral risk."""

    policy_id = "ortools_optimizer"
    policy_type = PolicyType.OPTIMIZER
    policy_version = "1.0.0"

    def __init__(self, solver_time_limit_sec: float = 0.25) -> None:
        self.solver_time_limit_sec = solver_time_limit_sec
        self.last_solve_ms: float = 0.0

    def select_action(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        _ = observation
        t0 = time.perf_counter()
        ctx: DecisionContext = info["context"]
        vehicle_ids: list[str] = info["vehicle_ids"]
        action = np.zeros(10, dtype=np.int64)
        orders = ctx.visible_orders
        if not orders:
            self.last_solve_ms = 0.0
            return action

        veh_map = {v.vehicle_id: v for v in ctx.vehicles}
        vehs = [veh_map[vid] for vid in vehicle_ids if vid in veh_map]
        n_orders = len(orders)
        n_vehs = len(vehs)
        overflow_code = len(vehicle_ids) + 1

        model = cp_model.CpModel()
        x: dict[tuple[int, int], cp_model.IntVar] = {}
        y: dict[int, cp_model.IntVar] = {}
        u = [model.new_bool_var(f"used_{j}") for j in range(n_vehs)]

        for i in range(n_orders):
            y[i] = model.new_bool_var(f"overflow_{i}")
            for j in range(n_vehs):
                x[(i, j)] = model.new_bool_var(f"x_{i}_{j}")

        # 1. Each order assigned to at most one vehicle or overflow
        for i in range(n_orders):
            model.add(sum(x[(i, j)] for j in range(n_vehs)) + y[i] <= 1)

        # 2. Vehicle availability, safety/hazmat, weight, volume, units, HOS constraints
        scale = 100
        for j, v in enumerate(vehs):
            if not v.available or v.broken_down:
                model.add(u[j] == 0)
                for i in range(n_orders):
                    model.add(x[(i, j)] == 0)
                continue

            for i, o in enumerate(orders):
                model.add(x[(i, j)] <= u[j])
                if (
                    o.requires_safety_segregation
                    or o.handling_class == HandlingClass.HAZMAT_BATTERY
                ) and not v.hazmat_certified:
                    model.add(x[(i, j)] == 0)

            model.add(
                sum(int(round(orders[i].total_weight_kg * scale)) * x[(i, j)] for i in range(n_orders))
                <= int(round(v.capacity_kg * scale))
            )
            model.add(
                sum(int(round(orders[i].total_volume_m3 * scale)) * x[(i, j)] for i in range(n_orders))
                <= int(round(v.capacity_m3 * scale))
            )
            model.add(
                sum(orders[i].quantity * x[(i, j)] for i in range(n_orders)) <= v.capacity_units
            )

            # Conservative upper bound matching ConstraintGuard's exact HOS rule:
            # first order pays 1.0x base, subsequent orders pay 0.45x base.
            est_mins = []
            for o in orders:
                d1 = ctx.lane_base_minutes.get((v.current_location_id, o.origin_id), 20.0)
                d2 = ctx.lane_base_minutes.get((o.origin_id, o.destination_hub_id), 20.0)
                est_mins.append(d1 + d2 + 15.0)
            max_first_overhead = int(math.ceil(0.55 * max(est_mins))) if est_mins else 0
            model.add(
                max_first_overhead * u[j]
                + sum(int(math.ceil(0.45 * est_mins[i])) * x[(i, j)] for i in range(n_orders))
                <= int(max(0.0, v.remaining_shift_minutes))
            )

        # 3. Objective: minimize dispatch fixed + deadhead + variable travel + overflow + SLA deferral risk
        obj_terms: list[cp_model.LinearExprT] = []
        for j, v in enumerate(vehs):
            # Fixed dispatch cost scaled x100
            obj_terms.append(int(round(12.0 * scale)) * u[j])
            for i, o in enumerate(orders):
                d_deadhead = ctx.lane_distances_km.get((v.current_location_id, o.origin_id), 20.0)
                d_line = ctx.lane_distances_km.get((o.origin_id, o.destination_hub_id), 20.0)
                hub_mismatch_pen = 0.0 if v.home_hub_id == o.destination_hub_id else 6.0
                var_cost = (0.45 * d_deadhead + d_line) * v.cost_per_km + hub_mismatch_pen
                obj_terms.append(int(round(var_cost * scale)) * x[(i, j)])

        for i, o in enumerate(orders):
            slack_hours = (o.promise_window_end_minute - ctx.timestamp_minute) / 60.0
            # Deferring gets increasingly expensive as deadline approaches
            urgency_weight = 16.0 * o.priority + (140.0 / max(0.5, slack_hours))
            if slack_hours < 2.0:
                urgency_weight += 180.0
            defer_var = 1 - sum(x[(i, j)] for j in range(n_vehs)) - y[i]
            obj_terms.append(int(round(urgency_weight * scale)) * defer_var)
            # Overflow 3PL cost (used only if fleet cannot meet tight SLA)
            overflow_cost = 46.0 + 1.6 * o.estimated_distance_km
            obj_terms.append(int(round(overflow_cost * scale)) * y[i])

        model.minimize(sum(obj_terms))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solver_time_limit_sec
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for i in range(n_orders):
                if solver.value(y[i]) == 1:
                    action[i] = overflow_code
                    continue
                for j in range(n_vehs):
                    if solver.value(x[(i, j)]) == 1:
                        action[i] = j + 1
                        break

        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        return action
