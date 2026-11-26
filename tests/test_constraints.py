"""Hard operational constraint guard tests (capacity, hazmat safety, breakdown, HOS)."""

from __future__ import annotations

from app.domain.entities import OrderAssignment, PublicOrderView, ResourceCalendar, Vehicle
from app.domain.enums import HandlingClass, OrderStatus
from app.simulation.constraints import ConstraintGuard, VehicleRuntimeStatus


def _make_order(
    order_id: str,
    weight_kg: float = 10.0,
    volume_m3: float = 0.05,
    qty: int = 1,
    hazmat: bool = False,
) -> PublicOrderView:
    return PublicOrderView(
        order_id=order_id,
        sku_id="SKU-TEST",
        category="electronics",
        quantity=qty,
        total_weight_kg=weight_kg,
        total_volume_m3=volume_m3,
        handling_class=HandlingClass.HAZMAT_BATTERY if hazmat else HandlingClass.STANDARD,
        requires_safety_segregation=hazmat,
        origin_id="ZONE-NW",
        destination_hub_id="HUB-NORTH",
        created_minute=60.0,
        promise_window_start_minute=90.0,
        promise_window_end_minute=540.0,
        reported_condition="opened_good",
        priority=3,
        unit_value_usd=150.0,
        estimated_distance_km=12.0,
        status=OrderStatus.PENDING,
    )


def test_constraint_guard_blocks_hazmat_capacity_breakdown_and_hos() -> None:
    guard = ConstraintGuard()
    cal = ResourceCalendar(calendar_id="cal_day", shift_start_hour=6, shift_end_hour=22)

    veh_safe = Vehicle(
        vehicle_id="V-STD",
        home_hub_id="HUB-NORTH",
        capacity_kg=50.0,
        capacity_m3=1.0,
        capacity_units=5,
        hazmat_certified=False,
    )
    veh_broken = Vehicle(
        vehicle_id="V-BRK",
        home_hub_id="HUB-NORTH",
        capacity_kg=200.0,
        capacity_m3=2.0,
        capacity_units=20,
        hazmat_certified=True,
    )
    veh_low_hos = Vehicle(
        vehicle_id="V-HOS",
        home_hub_id="HUB-NORTH",
        capacity_kg=200.0,
        capacity_m3=2.0,
        capacity_units=20,
        hazmat_certified=True,
    )

    vstates = {
        "V-STD": VehicleRuntimeStatus(vehicle=veh_safe, current_location_id="HUB-NORTH"),
        "V-BRK": VehicleRuntimeStatus(
            vehicle=veh_broken, current_location_id="HUB-NORTH", broken_down=True
        ),
        "V-HOS": VehicleRuntimeStatus(
            vehicle=veh_low_hos,
            current_location_id="HUB-NORTH",
            worked_minutes_today=510.0,
            max_shift_minutes=540.0,
        ),
    }

    orders = {
        "O-HAZ": _make_order("O-HAZ", hazmat=True),
        "O-HEAVY": _make_order("O-HEAVY", weight_kg=60.0),
        "O-BRK": _make_order("O-BRK"),
        "O-HOS": _make_order("O-HOS"),
        "O-OK": _make_order("O-OK", weight_kg=20.0),
    }
    proposed = [
        OrderAssignment(order_id="O-HAZ", vehicle_id="V-STD", action_type="dispatch"),
        OrderAssignment(order_id="O-HEAVY", vehicle_id="V-STD", action_type="dispatch"),
        OrderAssignment(order_id="O-BRK", vehicle_id="V-BRK", action_type="dispatch"),
        OrderAssignment(order_id="O-HOS", vehicle_id="V-HOS", action_type="dispatch"),
        OrderAssignment(order_id="O-OK", vehicle_id="V-STD", action_type="dispatch"),
    ]

    res = guard.validate_and_filter(
        sim_minute=600.0,  # 10:00 AM
        proposed=proposed,
        visible_orders=orders,
        vehicle_states=vstates,
        calendars={"cal_day": cal},
        hub_storage_free={"HUB-NORTH": 500},
        lane_minutes_fn=lambda a, b: 20.0,
    )

    assert res.violation_count == 4
    accepted_dispatch = [a for a in res.accepted_assignments if a.action_type == "dispatch"]
    assert len(accepted_dispatch) == 1
    assert accepted_dispatch[0].order_id == "O-OK"
    reasons_joined = " ".join(res.violation_reasons)
    assert "safety_hazmat_mismatch" in reasons_joined
    assert "weight_capacity_exceeded" in reasons_joined
    assert "vehicle_broken_down" in reasons_joined
    assert "driver_hos_exceeded" in reasons_joined
