"""Deterministic synthetic dataset generator for reverse-logistics operations."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime

import numpy as np

from app.domain.entities import (
    SKU,
    CapacityConstraint,
    Driver,
    InventoryPosition,
    Lane,
    Location,
    NetworkDataset,
    Order,
    Picker,
    ResourceCalendar,
    Robot,
    Vehicle,
)
from app.domain.enums import DispositionOutcome, HandlingClass, LocationType


def _euclidean_km(a: Location, b: Location) -> float:
    return math.hypot(a.x_km - b.x_km, a.y_km - b.y_km)


def generate_synthetic_dataset(
    seed: int = 42,
    horizon_minutes: float = 4320.0,
    orders_per_day: int = 110,
    dataset_id: str = "synthetic_reverse_v1",
) -> NetworkDataset:
    """Generate a reproducible, clearly marked synthetic e-commerce reverse-logistics dataset."""
    rng = np.random.default_rng(seed)

    calendars = [
        ResourceCalendar(
            calendar_id="cal_driver_day",
            shift_start_hour=6,
            shift_end_hour=22,
            working_days=[0, 1, 2, 3, 4, 5, 6],
            max_continuous_hours=10.0,
        ),
        ResourceCalendar(
            calendar_id="cal_hub_shift",
            shift_start_hour=6,
            shift_end_hour=23,
            working_days=[0, 1, 2, 3, 4, 5, 6],
            max_continuous_hours=12.0,
        ),
        ResourceCalendar(
            calendar_id="cal_standard",
            shift_start_hour=0,
            shift_end_hour=24,
            working_days=[0, 1, 2, 3, 4, 5, 6],
            max_continuous_hours=24.0,
        ),
    ]

    skus = [
        SKU(
            sku_id="SKU-APP-01",
            name="Winter Jacket",
            category="apparel",
            unit_weight_kg=1.2,
            unit_volume_m3=0.018,
            unit_value_usd=140.0,
            recovery_value_usd=98.0,
            handling_class=HandlingClass.STANDARD,
            inspection_minutes=6.0,
            restockable=True,
            refurbishable=True,
        ),
        SKU(
            sku_id="SKU-APP-02",
            name="Running Sneakers",
            category="footwear",
            unit_weight_kg=1.0,
            unit_volume_m3=0.015,
            unit_value_usd=115.0,
            recovery_value_usd=78.0,
            handling_class=HandlingClass.STANDARD,
            inspection_minutes=5.0,
            restockable=True,
            refurbishable=True,
        ),
        SKU(
            sku_id="SKU-ELC-01",
            name="Noise-Cancelling Headphones",
            category="electronics",
            unit_weight_kg=0.7,
            unit_volume_m3=0.010,
            unit_value_usd=260.0,
            recovery_value_usd=185.0,
            handling_class=HandlingClass.HAZMAT_BATTERY,
            inspection_minutes=14.0,
            restockable=True,
            refurbishable=True,
            refurb_kit_sku_id="KIT-ELC",
        ),
        SKU(
            sku_id="SKU-ELC-02",
            name="Cordless Handheld Vacuum",
            category="appliances",
            unit_weight_kg=3.1,
            unit_volume_m3=0.036,
            unit_value_usd=210.0,
            recovery_value_usd=135.0,
            handling_class=HandlingClass.HAZMAT_BATTERY,
            inspection_minutes=16.0,
            restockable=False,
            refurbishable=True,
            refurb_kit_sku_id="KIT-ELC",
        ),
        SKU(
            sku_id="SKU-HOM-01",
            name="Ceramic Espresso Maker",
            category="home_kitchen",
            unit_weight_kg=6.5,
            unit_volume_m3=0.065,
            unit_value_usd=320.0,
            recovery_value_usd=210.0,
            handling_class=HandlingClass.FRAGILE,
            inspection_minutes=15.0,
            restockable=True,
            refurbishable=True,
            refurb_kit_sku_id="KIT-HOM",
        ),
        SKU(
            sku_id="SKU-HOM-02",
            name="Smart Air Purifier",
            category="home_appliances",
            unit_weight_kg=5.2,
            unit_volume_m3=0.055,
            unit_value_usd=190.0,
            recovery_value_usd=120.0,
            handling_class=HandlingClass.FRAGILE,
            inspection_minutes=12.0,
            restockable=True,
            refurbishable=True,
        ),
    ]

    locations = [
        Location(
            location_id="HUB-NORTH",
            name="North Return & Triage Hub",
            location_type=LocationType.RETURN_HUB,
            x_km=0.0,
            y_km=8.0,
            dock_count=3,
            inspection_bays=4,
            robot_units=2,
            storage_capacity_units=1200,
            calendar_id="cal_hub_shift",
        ),
        Location(
            location_id="HUB-SOUTH",
            name="South Return & Triage Hub",
            location_type=LocationType.RETURN_HUB,
            x_km=4.0,
            y_km=-10.0,
            dock_count=3,
            inspection_bays=3,
            robot_units=2,
            storage_capacity_units=1000,
            calendar_id="cal_hub_shift",
        ),
        Location(
            location_id="ZONE-NW",
            name="Northwest Residential Zone",
            location_type=LocationType.CUSTOMER_ZONE,
            x_km=-18.0,
            y_km=19.0,
        ),
        Location(
            location_id="ZONE-NE",
            name="Northeast Tech Corridor",
            location_type=LocationType.CUSTOMER_ZONE,
            x_km=16.0,
            y_km=21.0,
        ),
        Location(
            location_id="ZONE-CORE",
            name="Downtown Dropoff Locker Cluster",
            location_type=LocationType.DROPOFF_POINT,
            x_km=-2.0,
            y_km=1.0,
        ),
        Location(
            location_id="ZONE-EAST",
            name="East Retail Parcel Shop",
            location_type=LocationType.DROPOFF_POINT,
            x_km=22.0,
            y_km=2.0,
        ),
        Location(
            location_id="ZONE-WEST",
            name="West Suburban Zone",
            location_type=LocationType.CUSTOMER_ZONE,
            x_km=-21.0,
            y_km=-6.0,
        ),
        Location(
            location_id="ZONE-SW",
            name="Southwest Border Zone",
            location_type=LocationType.CUSTOMER_ZONE,
            x_km=-14.0,
            y_km=-22.0,
        ),
        Location(
            location_id="ZONE-SE",
            name="Southeast Industrial Park",
            location_type=LocationType.DROPOFF_POINT,
            x_km=19.0,
            y_km=-18.0,
        ),
        Location(
            location_id="REFURB-CTR",
            name="Central Refurbishment Center",
            location_type=LocationType.REFURB_CENTER,
            x_km=2.0,
            y_km=0.0,
            dock_count=4,
            storage_capacity_units=2500,
        ),
    ]

    loc_map = {loc.location_id: loc for loc in locations}
    lanes: list[Lane] = []
    for orig in locations:
        for dest in locations:
            if orig.location_id == dest.location_id:
                continue
            dist = round(max(1.5, _euclidean_km(orig, dest) * 1.18), 2)
            base_mins = round(max(6.0, (dist / 42.0) * 60.0), 1)
            is_cross_border = "SW" in orig.location_id or "SW" in dest.location_id
            lanes.append(
                Lane(
                    lane_id=f"LANE-{orig.location_id}__{dest.location_id}",
                    origin_id=orig.location_id,
                    destination_id=dest.location_id,
                    distance_km=dist,
                    base_transit_minutes=base_mins,
                    cost_per_km=1.18,
                    fixed_dispatch_cost=9.0,
                    emission_kg_co2_per_km=0.27,
                    is_cross_border=is_cross_border,
                    active=True,
                )
            )

    vehicles = [
        Vehicle(
            vehicle_id="VEH-01",
            home_hub_id="HUB-NORTH",
            capacity_kg=180.0,
            capacity_m3=2.2,
            capacity_units=22,
            hazmat_certified=True,
            speed_kmph=45.0,
            cost_per_hour=34.0,
            cost_per_km=1.15,
            emission_kg_co2_per_km=0.27,
        ),
        Vehicle(
            vehicle_id="VEH-02",
            home_hub_id="HUB-NORTH",
            capacity_kg=160.0,
            capacity_m3=2.0,
            capacity_units=20,
            hazmat_certified=False,
            speed_kmph=48.0,
            cost_per_hour=30.0,
            cost_per_km=1.05,
            emission_kg_co2_per_km=0.24,
        ),
        Vehicle(
            vehicle_id="VEH-03",
            home_hub_id="HUB-NORTH",
            capacity_kg=220.0,
            capacity_m3=2.6,
            capacity_units=26,
            hazmat_certified=True,
            speed_kmph=42.0,
            cost_per_hour=36.0,
            cost_per_km=1.22,
            emission_kg_co2_per_km=0.31,
        ),
        Vehicle(
            vehicle_id="VEH-04",
            home_hub_id="HUB-SOUTH",
            capacity_kg=180.0,
            capacity_m3=2.2,
            capacity_units=22,
            hazmat_certified=True,
            speed_kmph=46.0,
            cost_per_hour=33.0,
            cost_per_km=1.12,
            emission_kg_co2_per_km=0.26,
        ),
        Vehicle(
            vehicle_id="VEH-05",
            home_hub_id="HUB-SOUTH",
            capacity_kg=150.0,
            capacity_m3=1.8,
            capacity_units=18,
            hazmat_certified=False,
            speed_kmph=50.0,
            cost_per_hour=29.0,
            cost_per_km=1.00,
            emission_kg_co2_per_km=0.21,
        ),
        Vehicle(
            vehicle_id="VEH-06",
            home_hub_id="HUB-SOUTH",
            capacity_kg=210.0,
            capacity_m3=2.5,
            capacity_units=25,
            hazmat_certified=False,
            speed_kmph=44.0,
            cost_per_hour=32.0,
            cost_per_km=1.10,
            emission_kg_co2_per_km=0.25,
        ),
    ]

    drivers = [
        Driver(driver_id=f"DRV-{i+1:02d}", assigned_vehicle_id=veh.vehicle_id, max_shift_minutes=540.0)
        for i, veh in enumerate(vehicles)
    ]

    pickers = [
        Picker(picker_id="INSP-N1", hub_id="HUB-NORTH", inspection_speed_multiplier=1.0),
        Picker(picker_id="INSP-N2", hub_id="HUB-NORTH", inspection_speed_multiplier=1.1),
        Picker(picker_id="INSP-N3", hub_id="HUB-NORTH", inspection_speed_multiplier=0.95),
        Picker(picker_id="INSP-N4", hub_id="HUB-NORTH", inspection_speed_multiplier=1.0),
        Picker(picker_id="INSP-S1", hub_id="HUB-SOUTH", inspection_speed_multiplier=1.0),
        Picker(picker_id="INSP-S2", hub_id="HUB-SOUTH", inspection_speed_multiplier=1.05),
        Picker(picker_id="INSP-S3", hub_id="HUB-SOUTH", inspection_speed_multiplier=0.95),
    ]

    robots = [
        Robot(robot_id="BOT-N1", hub_id="HUB-NORTH", max_payload_kg=50.0, move_minutes_per_task=2.0),
        Robot(robot_id="BOT-N2", hub_id="HUB-NORTH", max_payload_kg=50.0, move_minutes_per_task=2.0),
        Robot(robot_id="BOT-S1", hub_id="HUB-SOUTH", max_payload_kg=50.0, move_minutes_per_task=2.2),
        Robot(robot_id="BOT-S2", hub_id="HUB-SOUTH", max_payload_kg=50.0, move_minutes_per_task=2.2),
    ]

    hubs = [loc for loc in locations if loc.location_type == LocationType.RETURN_HUB]
    zones = [
        loc
        for loc in locations
        if loc.location_type in {LocationType.CUSTOMER_ZONE, LocationType.DROPOFF_POINT}
    ]

    constraints: list[CapacityConstraint] = []
    for veh in vehicles:
        constraints.append(
            CapacityConstraint(
                constraint_id=f"CONS-VEH-{veh.vehicle_id}",
                scope="vehicle",
                target_id=veh.vehicle_id,
                max_weight_kg=veh.capacity_kg,
                max_volume_m3=veh.capacity_m3,
                max_units=veh.capacity_units,
                require_hazmat_cert=not veh.hazmat_certified,
                hard=True,
            )
        )
    for drv in drivers:
        constraints.append(
            CapacityConstraint(
                constraint_id=f"CONS-HOS-{drv.driver_id}",
                scope="driver_hos",
                target_id=drv.assigned_vehicle_id,
                max_shift_minutes=drv.max_shift_minutes,
                hard=True,
            )
        )
    for hub in hubs:
        constraints.append(
            CapacityConstraint(
                constraint_id=f"CONS-DOCK-{hub.location_id}",
                scope="dock",
                target_id=hub.location_id,
                max_concurrent_slots=hub.dock_count,
                max_units=hub.storage_capacity_units,
                hard=True,
            )
        )
    constraints.append(
        CapacityConstraint(
            constraint_id="CONS-SLA-GLOBAL",
            scope="sla",
            target_id="GLOBAL",
            min_on_time_rate=0.88,
            hard=True,
        )
    )

    initial_inventory: list[InventoryPosition] = []
    for hub in hubs:
        for sku in skus:
            initial_inventory.append(
                InventoryPosition(
                    timestamp_minute=0.0,
                    location_id=hub.location_id,
                    sku_id=sku.sku_id,
                    on_hand_restock_units=25,
                    quarantined_units=0,
                    in_inspection_units=0,
                    refurb_kit_units=35,
                    backorder_units=0,
                    reorder_point=12,
                )
            )

    # Generate reverse pickup orders across horizon with morning/afternoon waves
    total_days = horizon_minutes / 1440.0
    n_orders = max(30, int(round(orders_per_day * total_days)))
    sku_probs = np.array([0.24, 0.22, 0.14, 0.12, 0.14, 0.14])
    zone_probs = np.array([0.14, 0.16, 0.20, 0.15, 0.13, 0.10, 0.12])
    reasons = ["size_or_fit", "defective", "damaged_in_transit", "changed_mind", "wrong_item"]
    conditions = ["unopened_new", "opened_good", "minor_wear", "damaged_box"]

    raw_timestamps: list[float] = []
    while len(raw_timestamps) < n_orders:
        day_idx = int(rng.integers(0, max(1, math.ceil(total_days))))
        # Peak arrival between 08:00 and 19:00
        hour = float(np.clip(rng.normal(13.5, 3.8), 6.0, 21.5))
        minute = day_idx * 1440.0 + hour * 60.0 + float(rng.uniform(0, 55.0))
        if minute < horizon_minutes - 180.0:
            raw_timestamps.append(round(minute, 2))
    raw_timestamps.sort()

    orders: list[Order] = []
    for idx, created_min in enumerate(raw_timestamps):
        sku = skus[int(rng.choice(len(skus), p=sku_probs))]
        origin = zones[int(rng.choice(len(zones), p=zone_probs))]
        # Nearest hub is the nominal destination hub
        nearest_hub = min(hubs, key=lambda h: _euclidean_km(loc_map[origin.location_id], h))
        qty = int(rng.choice([1, 1, 1, 2, 3]))
        priority = int(rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.30, 0.25, 0.15, 0.05]))
        # Promise SLA: 4 to 14 hours depending on priority
        sla_hours = max(4.0, 14.0 - 2.0 * (priority - 1) + float(rng.uniform(-1.0, 2.5)))
        promise_start = round(created_min + 30.0, 2)
        promise_end = round(min(horizon_minutes, created_min + sla_hours * 60.0), 2)
        if promise_end <= promise_start:
            promise_end = round(promise_start + 120.0, 2)

        cond = str(rng.choice(conditions, p=[0.30, 0.40, 0.20, 0.10]))
        if cond == "unopened_new" and sku.restockable:
            true_disp = DispositionOutcome.RESTOCK
        elif cond == "opened_good" and sku.restockable:
            true_disp = (
                DispositionOutcome.RESTOCK if rng.random() < 0.7 else DispositionOutcome.REFURBISH
            )
        elif sku.refurbishable and rng.random() < 0.72:
            true_disp = DispositionOutcome.REFURBISH
        elif rng.random() < 0.6:
            true_disp = DispositionOutcome.LIQUIDATE
        else:
            true_disp = DispositionOutcome.RECYCLE

        cancel_min: float | None = None
        if rng.random() < 0.04:
            cancel_min = round(created_min + float(rng.uniform(25.0, 180.0)), 2)

        orders.append(
            Order(
                order_id=f"RMA-{idx+1:05d}",
                sku_id=sku.sku_id,
                quantity=qty,
                origin_id=origin.location_id,
                destination_hub_id=nearest_hub.location_id,
                created_minute=created_min,
                promise_window_start_minute=promise_start,
                promise_window_end_minute=promise_end,
                return_reason=str(rng.choice(reasons)),
                reported_condition=cond,
                true_disposition=true_disp,
                priority=priority,
                requires_safety_segregation=(sku.handling_class == HandlingClass.HAZMAT_BATTERY),
                cancel_minute=cancel_min,
                is_synthetic=True,
            )
        )

    snapshot_payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "seed": seed,
            "n_orders": len(orders),
            "n_skus": len(skus),
            "n_locations": len(locations),
            "first_order": orders[0].model_dump(mode="json") if orders else {},
        },
        sort_keys=True,
    )
    snapshot_id = hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()[:16]

    return NetworkDataset(
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        is_synthetic=True,
        generated_at=datetime.now(UTC),
        skus=skus,
        locations=locations,
        lanes=lanes,
        orders=orders,
        initial_inventory=initial_inventory,
        vehicles=vehicles,
        drivers=drivers,
        pickers=pickers,
        robots=robots,
        calendars=calendars,
        constraints=constraints,
    )
