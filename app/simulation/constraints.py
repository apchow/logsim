"""Hard operational constraint guard preventing illegal assignments in the simulator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.domain.entities import (
    OrderAssignment,
    PublicOrderView,
    ResourceCalendar,
    Vehicle,
)
from app.domain.enums import HandlingClass


@dataclass
class VehicleRuntimeStatus:
    """Mutable runtime state of a vehicle tracked inside the simulator."""

    vehicle: Vehicle
    current_location_id: str
    busy_until_minute: float = 0.0
    worked_minutes_today: float = 0.0
    max_shift_minutes: float = 540.0
    broken_down: bool = False
    dispatched_routes: int = 0
    total_busy_minutes: float = 0.0

    def is_available_at(self, sim_minute: float, calendar: ResourceCalendar) -> bool:
        """Check availability, breakdown status, shift calendar, and remaining HOS."""
        if self.broken_down:
            return False
        if sim_minute < self.busy_until_minute - 1e-6:
            return False
        if not calendar.is_active(sim_minute):
            return False
        return self.worked_minutes_today < self.max_shift_minutes - 15.0


@dataclass
class ConstraintCheckResult:
    """Result of validating a policy's proposed assignments against hard constraints."""

    accepted_assignments: list[OrderAssignment]
    violation_count: int
    violation_reasons: list[str]


class ConstraintGuard:
    """Enforces capacity, HOS, safety/hazmat, dock/working-hours, and SLA rules."""

    def validate_and_filter(
        self,
        sim_minute: float,
        proposed: list[OrderAssignment],
        visible_orders: dict[str, PublicOrderView],
        vehicle_states: dict[str, VehicleRuntimeStatus],
        calendars: dict[str, ResourceCalendar],
        hub_storage_free: dict[str, int],
        lane_minutes_fn: Callable[[str, str], float],
    ) -> ConstraintCheckResult:
        """Filter proposed assignments; block infeasible actions and record every violation."""
        accepted: list[OrderAssignment] = []
        violations = 0
        reasons: list[str] = []

        # Track cumulative weight, volume, units, and route minutes per vehicle for this epoch
        batch_weight: dict[str, float] = {vid: 0.0 for vid in vehicle_states}
        batch_volume: dict[str, float] = {vid: 0.0 for vid in vehicle_states}
        batch_units: dict[str, int] = {vid: 0 for vid in vehicle_states}
        batch_min_est: dict[str, float] = {}
        batch_045_sum: dict[str, float] = {vid: 0.0 for vid in vehicle_states}
        batch_hub_units: dict[str, int] = dict(hub_storage_free)
        seen_orders: set[str] = set()

        for item in proposed:
            if item.order_id in seen_orders:
                continue
            seen_orders.add(item.order_id)

            order = visible_orders.get(item.order_id)
            if order is None:
                # Assigning an order that is not visible (either nonexistent or empty slot)
                if item.action_type != "defer":
                    violations += 1
                    reasons.append(f"nonexistent_or_future_order:{item.order_id}")
                continue

            if item.action_type == "defer":
                accepted.append(item)
                continue

            if item.action_type == "overflow_carrier":
                hub_free = batch_hub_units.get(order.destination_hub_id, 0)
                if hub_free < order.quantity:
                    violations += 1
                    reasons.append(f"hub_storage_full:{order.destination_hub_id}:{order.order_id}")
                    accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                else:
                    batch_hub_units[order.destination_hub_id] = hub_free - order.quantity
                    accepted.append(item)
                continue

            # Dispatch to specific vehicle
            if item.vehicle_id is None or item.vehicle_id not in vehicle_states:
                violations += 1
                reasons.append(f"unknown_vehicle:{item.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            vst = vehicle_states[item.vehicle_id]
            veh = vst.vehicle
            cal = calendars.get(veh.calendar_id) or next(iter(calendars.values()))

            if not vst.is_available_at(sim_minute, cal):
                violations += 1
                why = "broken_down" if vst.broken_down else "unavailable_or_off_shift"
                reasons.append(f"vehicle_{why}:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            # Safety segregation rule: hazmat battery returns require certified vehicle
            if (
                order.requires_safety_segregation
                or order.handling_class == HandlingClass.HAZMAT_BATTERY
            ) and not veh.hazmat_certified:
                violations += 1
                reasons.append(f"safety_hazmat_mismatch:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            # Capacity checks (weight, volume, units)
            new_w = batch_weight[veh.vehicle_id] + order.total_weight_kg
            new_v = batch_volume[veh.vehicle_id] + order.total_volume_m3
            new_u = batch_units[veh.vehicle_id] + order.quantity
            if new_w > veh.capacity_kg + 1e-6:
                violations += 1
                reasons.append(f"weight_capacity_exceeded:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue
            if new_v > veh.capacity_m3 + 1e-6:
                violations += 1
                reasons.append(f"volume_capacity_exceeded:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue
            if new_u > veh.capacity_units:
                violations += 1
                reasons.append(f"unit_capacity_exceeded:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            # Hub storage limit check
            hub_free = batch_hub_units.get(order.destination_hub_id, 0)
            if hub_free < order.quantity:
                violations += 1
                reasons.append(f"hub_storage_full:{order.destination_hub_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            # Order-invariant batch HOS check: 1.0x for shortest trunk + 0.45x per additional stop
            est_leg_min = float(
                lane_minutes_fn(vst.current_location_id, order.origin_id)
                + lane_minutes_fn(order.origin_id, order.destination_hub_id)
                + 15.0
            )
            new_min_est = (
                est_leg_min
                if batch_units[veh.vehicle_id] == 0
                else min(batch_min_est[veh.vehicle_id], est_leg_min)
            )
            new_045_sum = batch_045_sum[veh.vehicle_id] + 0.45 * est_leg_min
            total_batch_min = 0.55 * new_min_est + new_045_sum
            if vst.worked_minutes_today + total_batch_min > vst.max_shift_minutes:
                violations += 1
                reasons.append(f"driver_hos_exceeded:{veh.vehicle_id}:{order.order_id}")
                accepted.append(OrderAssignment(order_id=order.order_id, action_type="defer"))
                continue

            # Accept valid assignment
            batch_weight[veh.vehicle_id] = new_w
            batch_volume[veh.vehicle_id] = new_v
            batch_units[veh.vehicle_id] = new_u
            batch_min_est[veh.vehicle_id] = new_min_est
            batch_045_sum[veh.vehicle_id] = new_045_sum
            batch_hub_units[order.destination_hub_id] = hub_free - order.quantity
            accepted.append(item)

        return ConstraintCheckResult(
            accepted_assignments=accepted,
            violation_count=violations,
            violation_reasons=reasons,
        )
