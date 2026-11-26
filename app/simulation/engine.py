"""SimPy discrete-event engine for reverse-logistics pickup and hub triage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import simpy

from app.config import SimulationSettings
from app.domain.entities import (
    SKU,
    InventoryPosition,
    Lane,
    Location,
    NetworkDataset,
    Order,
    PolicyDecision,
    PublicOrderView,
    ResourceCalendar,
    Scenario,
)
from app.domain.enums import (
    DispositionOutcome,
    EventType,
    LocationType,
    OrderStatus,
    OverlayType,
)
from app.routing.adapters import (
    GeoPoint,
    RouteMatrix,
    compute_matrix_sha256,
    get_route_matrix,
)
from app.simulation.constraints import ConstraintGuard, VehicleRuntimeStatus
from app.simulation.disruptions import ActiveDisruptionState, apply_static_scenario_overlays
from app.simulation.event_log import EventLog


@dataclass(frozen=True)
class HubCausalView:
    """Leakage-free state of a return hub visible to policies at timestamp t."""

    hub_id: str
    total_docks: int
    docks_in_use: int
    total_inspection_bays: int
    inspection_in_use: int
    quarantine_units: int
    restock_units: int
    refurb_kit_units: int
    storage_capacity_units: int


@dataclass(frozen=True)
class VehicleCausalView:
    """Leakage-free state of a pickup vehicle visible to policies at timestamp t."""

    vehicle_id: str
    home_hub_id: str
    current_location_id: str
    available: bool
    broken_down: bool
    hazmat_certified: bool
    capacity_kg: float
    capacity_m3: float
    capacity_units: int
    remaining_shift_minutes: float
    busy_until_minute: float
    cost_per_km: float


@dataclass(frozen=True)
class DecisionContext:
    """Strictly causal decision context exposing ONLY state known at timestamp_minute."""

    timestamp_minute: float
    epoch_index: int
    horizon_minutes: float
    visible_orders: tuple[PublicOrderView, ...]
    total_pending_backlog: int
    vehicles: tuple[VehicleCausalView, ...]
    hubs: tuple[HubCausalView, ...]
    active_disruptions: tuple[str, ...]
    lane_distances_km: dict[tuple[str, str], float]
    lane_base_minutes: dict[tuple[str, str], float]


@dataclass
class HubRuntime:
    """SimPy resources and inventory trackers for a return hub."""

    location: Location
    dock_res: simpy.Resource
    inspection_res: simpy.Resource
    robot_res: simpy.Resource
    inventory: dict[str, InventoryPosition]
    dock_busy_minutes: float = 0.0
    inspection_busy_minutes: float = 0.0
    storage_used_units: int = 0


@dataclass
class StepMetricsDelta:
    """Incremental outcome counters emitted over a single Gymnasium decision step."""

    completed_orders: int = 0
    on_time_orders: int = 0
    late_orders: int = 0
    late_minutes: float = 0.0
    transport_cost: float = 0.0
    processing_cost: float = 0.0
    holding_cost: float = 0.0
    penalty_cost: float = 0.0
    recovery_value: float = 0.0
    empty_miles_km: float = 0.0
    total_miles_km: float = 0.0
    carbon_kg_co2: float = 0.0
    constraint_violations: int = 0
    stockouts: int = 0


@dataclass
class OrderRuntimeRecord:
    """Internal tracking record for an order (ground-truth private until realized)."""

    order: Order
    status: OrderStatus = OrderStatus.PENDING
    assigned_vehicle_id: str | None = None
    completed_minute: float | None = None


class ReverseLogisticsSimulator:
    """SimPy discrete-event simulator with zero-leakage causal context and hard constraints."""

    def __init__(
        self,
        dataset: NetworkDataset,
        scenario: Scenario,
        seed: int,
        sim_settings: SimulationSettings | None = None,
        run_id: str | None = None,
        routing_provider: str | None = None,
        routing_profile: str = "driving",
        route_matrix: RouteMatrix | None = None,
        route_cache_path: str | Path | None = None,
    ) -> None:
        self.base_dataset = dataset
        self.scenario = scenario
        self.seed = seed
        self.settings = sim_settings or SimulationSettings(
            horizon_minutes=scenario.horizon_minutes,
            decision_interval_minutes=scenario.decision_interval_minutes,
        )
        self.run_id = run_id or f"run-{uuid4().hex[:10]}"
        self.rng = np.random.default_rng(seed)

        # Apply only static dataset overlays (demand multiplier, mix shift, promise tightening)
        self.dataset = apply_static_scenario_overlays(dataset, scenario, seed)

        self.env = simpy.Environment()
        self.event_log = EventLog(
            run_id=self.run_id,
            scenario_id=scenario.scenario_id,
            seed=seed,
            synthetic=self.dataset.is_synthetic,
        )
        self.disruptions = ActiveDisruptionState()
        self.guard = ConstraintGuard()

        self.skus: dict[str, SKU] = {s.sku_id: s for s in self.dataset.skus}
        self.locations: dict[str, Location] = {
            loc.location_id: loc for loc in self.dataset.locations
        }
        self.calendars: dict[str, ResourceCalendar] = {
            c.calendar_id: c for c in self.dataset.calendars
        }
        self.lanes: dict[tuple[str, str], Lane] = {
            (lane.origin_id, lane.destination_id): lane for lane in self.dataset.lanes
        }
        self.lane_distances: dict[tuple[str, str], float] = {
            k: v.distance_km for k, v in self.lanes.items()
        }
        self.lane_base_minutes: dict[tuple[str, str], float] = {
            k: v.base_transit_minutes for k, v in self.lanes.items()
        }
        self.routing_provider, self.route_matrix_sha256 = self._init_routing_matrix(
            routing_provider=routing_provider,
            routing_profile=routing_profile,
            route_matrix=route_matrix,
            route_cache_path=route_cache_path,
        )

        # Hubs & SimPy resources (supports return hubs as well as depot/hub/warehouse bundles)
        self.hubs: dict[str, HubRuntime] = {}
        hub_types = {
            LocationType.RETURN_HUB,
            LocationType.HUB,
            LocationType.DEPOT,
            LocationType.WAREHOUSE,
        }
        for loc in self.dataset.locations:
            if loc.location_type in hub_types:
                inv_map = {
                    inv.sku_id: inv.model_copy(deep=True)
                    for inv in self.dataset.initial_inventory
                    if inv.location_id == loc.location_id
                }
                for sku_id in self.skus:
                    if sku_id not in inv_map:
                        inv_map[sku_id] = InventoryPosition(
                            timestamp_minute=0.0,
                            location_id=loc.location_id,
                            sku_id=sku_id,
                        )
                self.hubs[loc.location_id] = HubRuntime(
                    location=loc,
                    dock_res=simpy.Resource(self.env, capacity=max(1, loc.dock_count)),
                    inspection_res=simpy.Resource(self.env, capacity=max(1, loc.inspection_bays or 3)),
                    robot_res=simpy.Resource(self.env, capacity=max(1, loc.robot_units or 2)),
                    inventory=inv_map,
                )

        # Vehicles & Drivers
        drv_by_veh = {d.assigned_vehicle_id: d for d in self.dataset.drivers}
        self.vehicles: dict[str, VehicleRuntimeStatus] = {}
        for veh in self.dataset.vehicles:
            drv = drv_by_veh.get(veh.vehicle_id)
            self.vehicles[veh.vehicle_id] = VehicleRuntimeStatus(
                vehicle=veh,
                current_location_id=veh.home_hub_id,
                max_shift_minutes=drv.max_shift_minutes if drv else 540.0,
            )

        # Order runtime state: ONLY orders with created_minute <= now are added to visible_pending
        self._all_orders: dict[str, OrderRuntimeRecord] = {
            o.order_id: OrderRuntimeRecord(order=o) for o in self.dataset.orders
        }
        self._visible_pending_ids: list[str] = []
        self._step_delta = StepMetricsDelta()
        self._cumulative = StepMetricsDelta()
        self._epoch_index = 0
        self._last_sim_time = 0.0
        self._last_day_reset = -1
        self._finalized = False

        # Register SimPy background processes
        self._schedule_background_processes()

    def _init_routing_matrix(
        self,
        routing_provider: str | None,
        routing_profile: str,
        route_matrix: RouteMatrix | None,
        route_cache_path: str | Path | None,
    ) -> tuple[str, str]:
        """Populate lane matrices from cached RouteMatrix when configured and return provenance hash."""
        active_matrix = route_matrix
        if active_matrix is None and routing_provider is not None:
            ordered_locs = sorted(self.locations.values(), key=lambda loc: loc.location_id)
            pts = [
                GeoPoint(location_id=loc.location_id, x_km=loc.x_km, y_km=loc.y_km)
                for loc in ordered_locs
            ]
            active_matrix = get_route_matrix(
                origins=pts,
                destinations=pts,
                provider=routing_provider,
                profile=routing_profile,
                cache_path=route_cache_path,
            )

        if active_matrix is not None:
            dist_map, dur_map = active_matrix.to_lane_maps()
            self.lane_distances.update(dist_map)
            self.lane_base_minutes.update(dur_map)
            return routing_provider or active_matrix.provider, active_matrix.matrix_sha256

        loc_ids = sorted(self.locations.keys())
        dists = [[self._lane_distance(o, d) for d in loc_ids] for o in loc_ids]
        durs = [[self._lane_base_minutes(o, d) for d in loc_ids] for o in loc_ids]
        return "dataset_lanes", compute_matrix_sha256(dists, durs)

    def _lane_distance(self, orig: str, dest: str) -> float:
        if orig == dest:
            return 0.0
        dist = self.lane_distances.get((orig, dest))
        if dist is not None:
            return dist
        a, b = self.locations[orig], self.locations[dest]
        return round(max(1.5, math.hypot(a.x_km - b.x_km, a.y_km - b.y_km) * 1.18), 2)

    def _lane_base_minutes(self, orig: str, dest: str) -> float:
        if orig == dest:
            return 0.0
        mins = self.lane_base_minutes.get((orig, dest))
        if mins is not None:
            return mins
        return max(5.0, (self._lane_distance(orig, dest) / 42.0) * 60.0)

    def _effective_transit_minutes(self, orig: str, dest: str, speed_kmph: float) -> float:
        if orig == dest:
            return 0.0
        dist = self._lane_distance(orig, dest)
        base = (dist / max(10.0, speed_kmph)) * 60.0
        mult = self.disruptions.transit_time_multiplier
        lane = self.lanes.get((orig, dest))
        if lane is not None and lane.is_cross_border:
            mult *= self.disruptions.cross_border_multiplier
        noise = float(self.rng.uniform(0.92, 1.12))
        return max(3.0, base * mult * noise)

    def _schedule_background_processes(self) -> None:
        # 1. Order arrivals & cancellations
        for rec in sorted(self._all_orders.values(), key=lambda r: (r.order.created_minute, r.order.order_id)):
            self.env.process(self._order_lifecycle_proc(rec))

        # 2. Daily driver HOS resets & vehicle_available events
        self.env.process(self._daily_shift_proc())

        # 3. Dynamic scenario disruptions
        for idx, overlay in enumerate(self.scenario.overlays):
            if overlay.overlay_type in {
                OverlayType.CARRIER_DELAY,
                OverlayType.LEAD_TIME_INCREASE,
                OverlayType.WAREHOUSE_CAPACITY_REDUCTION,
                OverlayType.LABOR_ABSENCE,
                OverlayType.VEHICLE_BREAKDOWN,
                OverlayType.PORT_CUSTOMS_DELAY,
                OverlayType.WEATHER_DISRUPTION,
                OverlayType.SKU_AVAILABILITY_SHOCK,
            }:
                self.env.process(self._disruption_proc(idx, overlay))

        # 4. Periodic forward-restock consumption & holding cost ticks
        self.env.process(self._inventory_consumption_proc())

    def _order_lifecycle_proc(self, rec: OrderRuntimeRecord) -> Any:
        order = rec.order
        if order.created_minute > 0:
            yield self.env.timeout(order.created_minute)
        self._visible_pending_ids.append(order.order_id)
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.ORDER_CREATED,
            entity_id=order.order_id,
            entity_type="order",
            location_id=order.origin_id,
            order_id=order.order_id,
            payload={
                "sku_id": order.sku_id,
                "quantity": order.quantity,
                "destination_hub_id": order.destination_hub_id,
                "promise_window_end_minute": order.promise_window_end_minute,
                "priority": order.priority,
            },
        )
        if order.cancel_minute is not None:
            wait = max(0.0, order.cancel_minute - self.env.now)
            yield self.env.timeout(wait)
            if rec.status == OrderStatus.PENDING:
                rec.status = OrderStatus.CANCELLED
                if order.order_id in self._visible_pending_ids:
                    self._visible_pending_ids.remove(order.order_id)
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.ORDER_CANCELLED,
                    entity_id=order.order_id,
                    entity_type="order",
                    location_id=order.origin_id,
                    order_id=order.order_id,
                    payload={"reason": "customer_cancelled_before_pickup"},
                )

    def _daily_shift_proc(self) -> Any:
        horizon = self.settings.horizon_minutes
        day = 0
        while day * 1440.0 < horizon:
            shift_start = day * 1440.0 + 360.0  # 06:00
            if shift_start > self.env.now:
                yield self.env.timeout(shift_start - self.env.now)
            if self.env.now >= horizon:
                break
            for vid, vst in self.vehicles.items():
                vst.worked_minutes_today = 0.0
                if not vst.broken_down:
                    self.event_log.append(
                        timestamp_minute=self.env.now,
                        event_type=EventType.VEHICLE_AVAILABLE,
                        entity_id=vid,
                        entity_type="vehicle",
                        location_id=vst.current_location_id,
                        resource_id=vid,
                        payload={"shift_day": day, "max_shift_minutes": vst.max_shift_minutes},
                    )
            day += 1

    def _disruption_proc(self, idx: int, overlay: Any) -> Any:
        start = float(overlay.start_minute)
        end = float(
            overlay.end_minute if overlay.end_minute is not None else self.settings.horizon_minutes
        )
        if start > self.env.now:
            yield self.env.timeout(start - self.env.now)

        label = f"{overlay.overlay_type}:{overlay.target_id or 'network'}:{idx}"
        self.disruptions.active_overlay_labels.append(label)
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.DISRUPTION_STARTED,
            entity_id=label,
            entity_type="disruption",
            location_id=overlay.target_id,
            payload={"overlay_type": str(overlay.overlay_type), "value": overlay.value},
        )

        held_dock_reqs: list[tuple[simpy.Resource, Any]] = []
        held_insp_reqs: list[tuple[simpy.Resource, Any]] = []
        broken_vid: str | None = None

        if overlay.overlay_type in {OverlayType.CARRIER_DELAY, OverlayType.WEATHER_DISRUPTION}:
            self.disruptions.transit_time_multiplier *= max(1.0, float(overlay.value))
        elif overlay.overlay_type == OverlayType.PORT_CUSTOMS_DELAY:
            self.disruptions.cross_border_multiplier *= max(1.0, float(overlay.value))
        elif overlay.overlay_type == OverlayType.LEAD_TIME_INCREASE:
            self.disruptions.inspection_time_multiplier *= max(1.0, float(overlay.value))
        elif overlay.overlay_type == OverlayType.VEHICLE_BREAKDOWN:
            target = overlay.target_id or next(iter(self.vehicles.keys()))
            if target in self.vehicles:
                self.vehicles[target].broken_down = True
                self.disruptions.broken_vehicles.add(target)
                broken_vid = target
        elif overlay.overlay_type == OverlayType.WAREHOUSE_CAPACITY_REDUCTION:
            targets = [overlay.target_id] if overlay.target_id in self.hubs else list(self.hubs)
            for hid in targets:
                hub = self.hubs[hid]
                dock_cap = int(hub.dock_res.capacity)
                drop_docks = min(
                    dock_cap - 1, max(1, int(round(dock_cap * overlay.value)))
                )
                for _ in range(drop_docks):
                    req = hub.dock_res.request()
                    yield req
                    held_dock_reqs.append((hub.dock_res, req))
        elif overlay.overlay_type == OverlayType.LABOR_ABSENCE:
            for hub in self.hubs.values():
                insp_cap = int(hub.inspection_res.capacity)
                drop_insp = min(
                    insp_cap - 1,
                    max(1, int(round(insp_cap * overlay.value))),
                )
                for _ in range(drop_insp):
                    req = hub.inspection_res.request()
                    yield req
                    held_insp_reqs.append((hub.inspection_res, req))
        elif overlay.overlay_type == OverlayType.SKU_AVAILABILITY_SHOCK:
            self.disruptions.sku_shock_active = True
            for hub in self.hubs.values():
                for inv in hub.inventory.values():
                    inv.refurb_kit_units = max(0, int(inv.refurb_kit_units * (1.0 - min(0.95, overlay.value))))

        duration = max(0.0, end - self.env.now)
        if duration > 0:
            yield self.env.timeout(duration)

        # Revert disruption
        if overlay.overlay_type in {OverlayType.CARRIER_DELAY, OverlayType.WEATHER_DISRUPTION}:
            self.disruptions.transit_time_multiplier /= max(1.0, float(overlay.value))
        elif overlay.overlay_type == OverlayType.PORT_CUSTOMS_DELAY:
            self.disruptions.cross_border_multiplier /= max(1.0, float(overlay.value))
        elif overlay.overlay_type == OverlayType.LEAD_TIME_INCREASE:
            self.disruptions.inspection_time_multiplier /= max(1.0, float(overlay.value))
        elif broken_vid is not None:
            self.vehicles[broken_vid].broken_down = False
            self.disruptions.broken_vehicles.discard(broken_vid)
        elif overlay.overlay_type == OverlayType.SKU_AVAILABILITY_SHOCK:
            self.disruptions.sku_shock_active = False

        for res, req in held_dock_reqs + held_insp_reqs:
            res.release(req)

        if label in self.disruptions.active_overlay_labels:
            self.disruptions.active_overlay_labels.remove(label)
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.DISRUPTION_ENDED,
            entity_id=label,
            entity_type="disruption",
            location_id=overlay.target_id,
            payload={"overlay_type": str(overlay.overlay_type)},
        )

    def _inventory_consumption_proc(self) -> Any:
        interval = 240.0  # every 4 hours, forward fulfillment draws restocked SKUs
        while self.env.now + interval < self.settings.horizon_minutes:
            yield self.env.timeout(interval)
            for hid, hub in self.hubs.items():
                total_held = 0
                for sku_id, inv in hub.inventory.items():
                    demand = int(self.rng.poisson(1.8 if not self.disruptions.sku_shock_active else 4.2))
                    if demand > 0:
                        consumed = min(inv.on_hand_restock_units, demand)
                        shortfall = demand - consumed
                        inv.on_hand_restock_units -= consumed
                        if consumed > 0:
                            self.event_log.append(
                                timestamp_minute=self.env.now,
                                event_type=EventType.INVENTORY_CONSUMED,
                                entity_id=f"{hid}:{sku_id}",
                                entity_type="inventory",
                                location_id=hid,
                                payload={"sku_id": sku_id, "units": consumed, "channel": "forward_restock"},
                            )
                        if shortfall > 0:
                            inv.backorder_units += shortfall
                            self._step_delta.stockouts += 1
                            self._cumulative.stockouts += 1
                            penalty = shortfall * self.settings.stockout_penalty_per_unit
                            self._step_delta.penalty_cost += penalty
                            self._cumulative.penalty_cost += penalty
                            self.event_log.append(
                                timestamp_minute=self.env.now,
                                event_type=EventType.STOCKOUT_OCCURRED,
                                entity_id=f"{hid}:{sku_id}",
                                entity_type="inventory",
                                location_id=hid,
                                payload={"sku_id": sku_id, "shortfall_units": shortfall},
                            )
                    total_held += inv.on_hand_restock_units + inv.quarantined_units + inv.in_inspection_units
                holding = total_held * self.settings.holding_cost_per_unit_hour * (interval / 60.0)
                self._step_delta.holding_cost += holding
                self._cumulative.holding_cost += holding

    def _resolve_hub_id(self, order: Order) -> str:
        """Resolve serving hub ID for both reverse (origin->hub) and forward (hub->dest) orders."""
        if order.destination_hub_id in self.hubs:
            return order.destination_hub_id
        if order.origin_id in self.hubs:
            return order.origin_id
        return next(iter(self.hubs))

    def build_causal_context(self) -> DecisionContext:
        """Build strictly causal DecisionContext at current simulation timestamp."""
        now = float(self.env.now)
        pending_views: list[PublicOrderView] = []
        for oid in self._visible_pending_ids:
            rec = self._all_orders[oid]
            if rec.status != OrderStatus.PENDING:
                continue
            o = rec.order
            if o.created_minute > now + 1e-9:
                # Defensive guard: never leak future orders
                continue
            sku = self.skus[o.sku_id]
            hub_id = self._resolve_hub_id(o)
            route_stop = o.origin_id if o.origin_id != hub_id else o.destination_hub_id
            pending_views.append(
                PublicOrderView(
                    order_id=o.order_id,
                    sku_id=o.sku_id,
                    category=sku.category,
                    quantity=o.quantity,
                    total_weight_kg=round(sku.unit_weight_kg * o.quantity, 3),
                    total_volume_m3=round(sku.unit_volume_m3 * o.quantity, 4),
                    handling_class=sku.handling_class,
                    requires_safety_segregation=o.requires_safety_segregation,
                    origin_id=route_stop,
                    destination_hub_id=hub_id,
                    created_minute=o.created_minute,
                    promise_window_start_minute=o.promise_window_start_minute,
                    promise_window_end_minute=o.promise_window_end_minute,
                    reported_condition=o.reported_condition,
                    priority=o.priority,
                    unit_value_usd=sku.unit_value_usd,
                    estimated_distance_km=self._lane_distance(route_stop, hub_id),
                    status=rec.status,
                )
            )

        # Deterministic urgency ordering: earliest promise deadline, then highest priority, then creation
        pending_views.sort(
            key=lambda p: (p.promise_window_end_minute, -p.priority, p.created_minute, p.order_id)
        )
        max_k = self.settings.max_visible_orders
        visible_slice = tuple(pending_views[:max_k])

        veh_views: list[VehicleCausalView] = []
        for vid in sorted(self.vehicles.keys()):
            vst = self.vehicles[vid]
            veh = vst.vehicle
            cal = self.calendars.get(veh.calendar_id) or next(iter(self.calendars.values()))
            veh_views.append(
                VehicleCausalView(
                    vehicle_id=vid,
                    home_hub_id=veh.home_hub_id,
                    current_location_id=vst.current_location_id,
                    available=vst.is_available_at(now, cal),
                    broken_down=vst.broken_down,
                    hazmat_certified=veh.hazmat_certified,
                    capacity_kg=veh.capacity_kg,
                    capacity_m3=veh.capacity_m3,
                    capacity_units=veh.capacity_units,
                    remaining_shift_minutes=max(0.0, vst.max_shift_minutes - vst.worked_minutes_today),
                    busy_until_minute=vst.busy_until_minute,
                    cost_per_km=veh.cost_per_km,
                )
            )

        hub_views: list[HubCausalView] = []
        for hid in sorted(self.hubs.keys()):
            hub = self.hubs[hid]
            hub_views.append(
                HubCausalView(
                    hub_id=hid,
                    total_docks=int(hub.dock_res.capacity),
                    docks_in_use=hub.dock_res.count,
                    total_inspection_bays=int(hub.inspection_res.capacity),
                    inspection_in_use=hub.inspection_res.count,
                    quarantine_units=sum(i.quarantined_units for i in hub.inventory.values()),
                    restock_units=sum(i.on_hand_restock_units for i in hub.inventory.values()),
                    refurb_kit_units=sum(i.refurb_kit_units for i in hub.inventory.values()),
                    storage_capacity_units=hub.location.storage_capacity_units,
                )
            )

        return DecisionContext(
            timestamp_minute=now,
            epoch_index=self._epoch_index,
            horizon_minutes=self.settings.horizon_minutes,
            visible_orders=visible_slice,
            total_pending_backlog=len(pending_views),
            vehicles=tuple(veh_views),
            hubs=tuple(hub_views),
            active_disruptions=tuple(self.disruptions.active_overlay_labels),
            lane_distances_km=self.lane_distances,
            lane_base_minutes=self.lane_base_minutes,
        )

    def advance_to_first_decision(self) -> DecisionContext:
        """Process t=0 initial events and return the initial causal decision context."""
        if self.env.now < 1e-4:
            self.env.run(until=1e-4)
        return self.build_causal_context()

    def apply_decision_and_step(self, decision: PolicyDecision) -> tuple[DecisionContext, StepMetricsDelta, bool]:
        """Validate decision via ConstraintGuard, spawn SimPy processes, and advance one epoch."""
        self._step_delta = StepMetricsDelta()
        ctx = self.build_causal_context()
        visible_map = {o.order_id: o for o in ctx.visible_orders}
        hub_free = {
            hid: max(0, h.location.storage_capacity_units - h.storage_used_units)
            for hid, h in self.hubs.items()
        }

        check = self.guard.validate_and_filter(
            sim_minute=self.env.now,
            proposed=decision.assignments,
            visible_orders=visible_map,
            vehicle_states=self.vehicles,
            calendars=self.calendars,
            hub_storage_free=hub_free,
            lane_minutes_fn=self._lane_base_minutes,
        )

        self._step_delta.constraint_violations += check.violation_count
        self._cumulative.constraint_violations += check.violation_count
        if check.violation_count > 0:
            penalty = check.violation_count * self.settings.violation_penalty_per_attempt
            self._step_delta.penalty_cost += penalty
            self._cumulative.penalty_cost += penalty
            for reason in check.violation_reasons:
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.EXCEPTION_RAISED,
                    entity_id=decision.decision_id,
                    entity_type="constraint_violation",
                    resource_id=decision.policy_id,
                    payload={"reason": reason, "policy_id": decision.policy_id},
                )

        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.POLICY_DECISION,
            entity_id=decision.decision_id,
            entity_type="policy_decision",
            resource_id=decision.policy_id,
            payload={
                "policy_id": decision.policy_id,
                "policy_type": str(decision.policy_type),
                "epoch_index": self._epoch_index,
                "observation_hash": decision.observation_hash or "",
                "accepted_count": len(check.accepted_assignments),
                "violation_count": check.violation_count,
                "solver_time_ms": round(decision.solver_time_ms, 3),
                "routing_provider": self.routing_provider,
                "matrix_sha256": self.route_matrix_sha256,
            },
        )

        # Group accepted vehicle dispatches by vehicle_id
        veh_batches: dict[str, list[OrderRuntimeRecord]] = {}
        for item in check.accepted_assignments:
            rec = self._all_orders.get(item.order_id)
            if rec is None or rec.status != OrderStatus.PENDING:
                continue
            if item.action_type == "overflow_carrier":
                rec.status = OrderStatus.ASSIGNED
                if rec.order.order_id in self._visible_pending_ids:
                    self._visible_pending_ids.remove(rec.order.order_id)
                self.env.process(self._overflow_carrier_proc(rec))
            elif item.action_type == "dispatch" and item.vehicle_id is not None:
                rec.status = OrderStatus.ASSIGNED
                rec.assigned_vehicle_id = item.vehicle_id
                if rec.order.order_id in self._visible_pending_ids:
                    self._visible_pending_ids.remove(rec.order.order_id)
                veh_batches.setdefault(item.vehicle_id, []).append(rec)

        for vid, batch in veh_batches.items():
            self.env.process(self._vehicle_pickup_route_proc(vid, batch))

        self._epoch_index += 1
        next_time = min(
            self.settings.horizon_minutes,
            self._epoch_index * self.settings.decision_interval_minutes,
        )
        if next_time > self.env.now:
            self.env.run(until=next_time)

        done = self.env.now >= self.settings.horizon_minutes - 1e-6
        if done and not self._finalized:
            self._finalize_horizon_penalties()
            self._finalized = True

        next_ctx = self.build_causal_context()
        return next_ctx, self._step_delta, done

    def _optimize_stop_sequence(
        self,
        start_loc: str,
        batch: list[OrderRuntimeRecord],
        target_hub_id: str,
    ) -> list[OrderRuntimeRecord]:
        """Order multi-stop pickup sequence via nearest-neighbor plus 2-opt refinement."""

        def _stop_loc(r: OrderRuntimeRecord) -> str:
            return r.order.origin_id if r.order.origin_id != target_hub_id else r.order.destination_hub_id

        remaining = list(batch)
        ordered: list[OrderRuntimeRecord] = []
        cursor = start_loc
        while remaining:
            nxt = min(remaining, key=lambda r: (self._lane_distance(cursor, _stop_loc(r)), r.order.order_id))
            ordered.append(nxt)
            cursor = _stop_loc(nxt)
            remaining.remove(nxt)

        if len(ordered) <= 2:
            return ordered

        def _tour_km(seq: list[OrderRuntimeRecord]) -> float:
            total = 0.0
            prev = start_loc
            for item in seq:
                loc = _stop_loc(item)
                total += self._lane_distance(prev, loc)
                prev = loc
            return total + self._lane_distance(prev, target_hub_id)

        best_km = _tour_km(ordered)
        improved = True
        while improved:
            improved = False
            n = len(ordered)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    candidate = ordered[:i] + list(reversed(ordered[i : j + 1])) + ordered[j + 1 :]
                    cand_km = _tour_km(candidate)
                    if cand_km + 1e-6 < best_km:
                        ordered = candidate
                        best_km = cand_km
                        improved = True
                        break
                if improved:
                    break
        return ordered

    def _vehicle_pickup_route_proc(self, vehicle_id: str, batch: list[OrderRuntimeRecord]) -> Any:
        vst = self.vehicles[vehicle_id]
        veh = vst.vehicle
        vst.dispatched_routes += 1
        start_time = self.env.now

        target_hub_id = self._resolve_hub_id(batch[0].order)

        def _stop_loc(r: OrderRuntimeRecord) -> str:
            return r.order.origin_id if r.order.origin_id != target_hub_id else r.order.destination_hub_id

        ordered_stops = self._optimize_stop_sequence(vst.current_location_id, batch, target_hub_id)

        first_stop = _stop_loc(ordered_stops[0])
        first_leg_km = self._lane_distance(vst.current_location_id, first_stop)
        # Empty/deadhead mileage is the positioning trip from current location to first pickup stop
        empty_km = first_leg_km
        total_km = 0.0
        cur_loc = vst.current_location_id

        # Estimate busy time conservatively until route ends
        vst.busy_until_minute = self.env.now + 30.0

        for rec in ordered_stops:
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.SHIPMENT_DISPATCHED,
                entity_id=rec.order.order_id,
                entity_type="shipment",
                location_id=cur_loc,
                order_id=rec.order.order_id,
                resource_id=vehicle_id,
                payload={
                    "vehicle_id": vehicle_id,
                    "origin_id": rec.order.origin_id,
                    "stop_count": len(ordered_stops),
                    "routing_provider": self.routing_provider,
                    "matrix_sha256": self.route_matrix_sha256,
                },
            )

        for rec in ordered_stops:
            stop_loc = (
                rec.order.origin_id
                if rec.order.origin_id != target_hub_id
                else rec.order.destination_hub_id
            )
            dest_zone = stop_loc
            leg_km = self._lane_distance(cur_loc, dest_zone)
            leg_min = self._effective_transit_minutes(cur_loc, dest_zone, veh.speed_kmph)
            if cur_loc != dest_zone:
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.VEHICLE_DEPARTED,
                    entity_id=vehicle_id,
                    entity_type="vehicle",
                    location_id=cur_loc,
                    order_id=rec.order.order_id,
                    resource_id=vehicle_id,
                    payload={"to_location": dest_zone, "distance_km": leg_km},
                )
                vst.busy_until_minute = max(vst.busy_until_minute, self.env.now + leg_min)
                yield self.env.timeout(leg_min)
                total_km += leg_km
                cur_loc = dest_zone
                vst.current_location_id = cur_loc
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.VEHICLE_ARRIVED,
                    entity_id=vehicle_id,
                    entity_type="vehicle",
                    location_id=cur_loc,
                    order_id=rec.order.order_id,
                    resource_id=vehicle_id,
                    payload={"at_location": cur_loc},
                )
            rec.status = OrderStatus.IN_TRANSIT
            # Dwell / loading time at stop
            load_min = 4.0 + 1.5 * rec.order.quantity
            vst.busy_until_minute = max(vst.busy_until_minute, self.env.now + load_min)
            yield self.env.timeout(load_min)

        # Return haul to destination return hub
        return_km = self._lane_distance(cur_loc, target_hub_id)
        return_min = self._effective_transit_minutes(cur_loc, target_hub_id, veh.speed_kmph)
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.VEHICLE_DEPARTED,
            entity_id=vehicle_id,
            entity_type="vehicle",
            location_id=cur_loc,
            resource_id=vehicle_id,
            payload={"to_location": target_hub_id, "distance_km": return_km, "batch_size": len(ordered_stops)},
        )
        vst.busy_until_minute = max(vst.busy_until_minute, self.env.now + return_min)
        yield self.env.timeout(return_min)
        total_km += return_km
        vst.current_location_id = target_hub_id
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.VEHICLE_ARRIVED,
            entity_id=vehicle_id,
            entity_type="vehicle",
            location_id=target_hub_id,
            resource_id=vehicle_id,
            payload={"at_location": target_hub_id},
        )

        # Hub dock acquisition & unloading
        hub = self.hubs[target_hub_id]
        with hub.dock_res.request() as dock_req:
            yield dock_req
            dock_start = self.env.now
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.DOCK_OCCUPIED,
                entity_id=target_hub_id,
                entity_type="dock",
                location_id=target_hub_id,
                resource_id=vehicle_id,
                payload={"batch_size": len(ordered_stops)},
            )
            unload_min = 6.0 + 2.0 * len(ordered_stops)
            yield self.env.timeout(unload_min)
            hub.dock_busy_minutes += self.env.now - dock_start
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.DOCK_RELEASED,
                entity_id=target_hub_id,
                entity_type="dock",
                location_id=target_hub_id,
                resource_id=vehicle_id,
                payload={"unload_minutes": unload_min},
            )

        route_minutes = self.env.now - start_time
        vst.worked_minutes_today += route_minutes
        vst.total_busy_minutes += route_minutes
        vst.busy_until_minute = self.env.now

        # Costs & emissions (slightly scaled by vehicle mass load factor)
        batch_weight_kg = sum(
            self.skus[r.order.sku_id].unit_weight_kg * r.order.quantity for r in ordered_stops
        )
        load_ratio = min(1.0, batch_weight_kg / max(1.0, veh.capacity_kg))
        transport_cost = (
            9.0
            + total_km * veh.cost_per_km
            + empty_km * self.settings.empty_mile_cost_factor
            + (route_minutes / 60.0) * veh.cost_per_hour
        )
        co2 = total_km * veh.emission_kg_co2_per_km * (1.0 + 0.08 * load_ratio)
        self._step_delta.transport_cost += transport_cost
        self._cumulative.transport_cost += transport_cost
        self._step_delta.total_miles_km += total_km
        self._cumulative.total_miles_km += total_km
        self._step_delta.empty_miles_km += empty_km
        self._cumulative.empty_miles_km += empty_km
        self._step_delta.carbon_kg_co2 += co2
        self._cumulative.carbon_kg_co2 += co2

        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.VEHICLE_AVAILABLE,
            entity_id=vehicle_id,
            entity_type="vehicle",
            location_id=target_hub_id,
            resource_id=vehicle_id,
            payload={
                "route_minutes": round(route_minutes, 2),
                "total_km": round(total_km, 2),
                "routing_provider": self.routing_provider,
                "matrix_sha256": self.route_matrix_sha256,
            },
        )

        # Deliver orders & start triage pipeline
        for rec in ordered_stops:
            self.env.process(self._hub_triage_proc(rec, target_hub_id))

    def _overflow_carrier_proc(self, rec: OrderRuntimeRecord) -> Any:
        order = rec.order
        hub_id_resolved = self._resolve_hub_id(order)
        route_stop = order.origin_id if order.origin_id != hub_id_resolved else order.destination_hub_id
        dist = self._lane_distance(route_stop, hub_id_resolved)
        transit = max(
            self.settings.overflow_carrier_transit_minutes,
            self._effective_transit_minutes(route_stop, hub_id_resolved, 35.0) * 1.5,
        )
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.SHIPMENT_DISPATCHED,
            entity_id=order.order_id,
            entity_type="shipment",
            location_id=order.origin_id,
            order_id=order.order_id,
            resource_id="3PL-OVERFLOW",
            payload={
                "mode": "overflow_carrier",
                "distance_km": dist,
                "routing_provider": self.routing_provider,
                "matrix_sha256": self.route_matrix_sha256,
            },
        )
        rec.status = OrderStatus.IN_TRANSIT
        yield self.env.timeout(transit)

        cost = self.settings.overflow_carrier_cost_per_order + dist * 1.65
        co2 = dist * 0.34
        self._step_delta.transport_cost += cost
        self._cumulative.transport_cost += cost
        self._step_delta.total_miles_km += dist
        self._cumulative.total_miles_km += dist
        self._step_delta.carbon_kg_co2 += co2
        self._cumulative.carbon_kg_co2 += co2

        hub = self.hubs[hub_id_resolved]
        with hub.dock_res.request() as dock_req:
            yield dock_req
            dock_start = self.env.now
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.DOCK_OCCUPIED,
                entity_id=hub.location.location_id,
                entity_type="dock",
                location_id=hub.location.location_id,
                order_id=order.order_id,
                resource_id="3PL-OVERFLOW",
                payload={"mode": "overflow_carrier"},
            )
            yield self.env.timeout(5.0)
            hub.dock_busy_minutes += self.env.now - dock_start
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.DOCK_RELEASED,
                entity_id=hub.location.location_id,
                entity_type="dock",
                location_id=hub.location.location_id,
                order_id=order.order_id,
                resource_id="3PL-OVERFLOW",
                payload={},
            )

        self.env.process(self._hub_triage_proc(rec, hub_id_resolved))

    def _hub_triage_proc(self, rec: OrderRuntimeRecord, hub_id: str) -> Any:
        order = rec.order
        sku = self.skus[order.sku_id]
        hub = self.hubs[hub_id]
        inv = hub.inventory[order.sku_id]

        # Delivered to hub dock -> quarantine inventory
        rec.status = OrderStatus.AT_DOCK
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.SHIPMENT_DELIVERED,
            entity_id=order.order_id,
            entity_type="shipment",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"on_time_arrival": self.env.now <= order.promise_window_end_minute},
        )
        inv.quarantined_units += order.quantity
        hub.storage_used_units += order.quantity
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.INVENTORY_RECEIVED,
            entity_id=f"{hub_id}:{order.sku_id}",
            entity_type="inventory",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"sku_id": order.sku_id, "units": order.quantity, "bucket": "quarantine"},
        )

        # Human inspector / picker triage task
        rec.status = OrderStatus.IN_INSPECTION
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.PICKER_TASK_CREATED,
            entity_id=f"INSP-{order.order_id}",
            entity_type="picker_task",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"sku_id": order.sku_id, "quantity": order.quantity},
        )
        with hub.inspection_res.request() as insp_req:
            yield insp_req
            insp_start = self.env.now
            inv.quarantined_units = max(0, inv.quarantined_units - order.quantity)
            inv.in_inspection_units += order.quantity
            insp_duration = (
                sku.inspection_minutes
                * max(1.0, order.quantity * 0.6)
                * self.disruptions.inspection_time_multiplier
            )
            yield self.env.timeout(insp_duration)
            hub.inspection_busy_minutes += self.env.now - insp_start
            inv.in_inspection_units = max(0, inv.in_inspection_units - order.quantity)

        # Ground-truth disposition revealed ONLY after physical inspection!
        disposition = order.true_disposition
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.PICKER_TASK_COMPLETED,
            entity_id=f"INSP-{order.order_id}",
            entity_type="picker_task",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"disposition": str(disposition)},
        )

        # Robot bin sortation task
        rec.status = OrderStatus.SORTING
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.ROBOT_TASK_CREATED,
            entity_id=f"BOT-{order.order_id}",
            entity_type="robot_task",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"target_disposition": str(disposition)},
        )
        with hub.robot_res.request() as bot_req:
            yield bot_req
            yield self.env.timeout(2.2)
        self.event_log.append(
            timestamp_minute=self.env.now,
            event_type=EventType.ROBOT_TASK_COMPLETED,
            entity_id=f"BOT-{order.order_id}",
            entity_type="robot_task",
            location_id=hub_id,
            order_id=order.order_id,
            payload={"target_disposition": str(disposition)},
        )

        # Disposition accounting & recovery credit
        proc_cost = 3.5 * order.quantity
        recovery = 0.0
        if disposition == DispositionOutcome.RESTOCK:
            inv.on_hand_restock_units += order.quantity
            recovery = sku.recovery_value_usd * order.quantity
        elif disposition == DispositionOutcome.REFURBISH:
            if inv.refurb_kit_units >= order.quantity:
                inv.refurb_kit_units -= order.quantity
                inv.on_hand_restock_units += order.quantity
                proc_cost += 11.0 * order.quantity
                recovery = sku.recovery_value_usd * 0.84 * order.quantity
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.INVENTORY_CONSUMED,
                    entity_id=f"{hub_id}:KIT-{order.sku_id}",
                    entity_type="inventory",
                    location_id=hub_id,
                    order_id=order.order_id,
                    payload={"kit_units": order.quantity},
                )
            else:
                self._step_delta.stockouts += 1
                self._cumulative.stockouts += 1
                proc_cost += 6.0 * order.quantity
                recovery = sku.recovery_value_usd * 0.35 * order.quantity
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.STOCKOUT_OCCURRED,
                    entity_id=f"{hub_id}:KIT-{order.sku_id}",
                    entity_type="inventory",
                    location_id=hub_id,
                    order_id=order.order_id,
                    payload={"reason": "missing_refurb_kit"},
                )
        elif disposition == DispositionOutcome.LIQUIDATE:
            hub.storage_used_units = max(0, hub.storage_used_units - order.quantity)
            recovery = sku.recovery_value_usd * 0.42 * order.quantity
        else:
            hub.storage_used_units = max(0, hub.storage_used_units - order.quantity)
            proc_cost += 4.0 * order.quantity
            recovery = sku.recovery_value_usd * 0.12 * order.quantity

        rec.status = OrderStatus.COMPLETED
        rec.completed_minute = self.env.now

        # SLA check based on delivery/triage completion against promise window
        on_time = self.env.now <= order.promise_window_end_minute + 60.0
        late_mins = max(0.0, self.env.now - (order.promise_window_end_minute + 60.0))
        late_penalty = (late_mins / 60.0) * self.settings.late_penalty_per_hour

        self._step_delta.completed_orders += 1
        self._cumulative.completed_orders += 1
        if on_time:
            self._step_delta.on_time_orders += 1
            self._cumulative.on_time_orders += 1
        else:
            self._step_delta.late_orders += 1
            self._cumulative.late_orders += 1
            self._step_delta.late_minutes += late_mins
            self._cumulative.late_minutes += late_mins
            self._step_delta.penalty_cost += late_penalty
            self._cumulative.penalty_cost += late_penalty
            self.event_log.append(
                timestamp_minute=self.env.now,
                event_type=EventType.EXCEPTION_RAISED,
                entity_id=order.order_id,
                entity_type="sla_breach",
                location_id=hub_id,
                order_id=order.order_id,
                payload={"late_minutes": round(late_mins, 2)},
            )

        self._step_delta.processing_cost += proc_cost
        self._cumulative.processing_cost += proc_cost
        self._step_delta.recovery_value += recovery
        self._cumulative.recovery_value += recovery

    def _finalize_horizon_penalties(self) -> None:
        """Apply backlog penalty for any non-cancelled orders left unserved at horizon end."""
        for rec in self._all_orders.values():
            if (
                rec.status in {OrderStatus.PENDING, OrderStatus.ASSIGNED, OrderStatus.IN_TRANSIT}
                and rec.order.created_minute <= self.settings.horizon_minutes - 60.0
            ):
                pen = self.settings.unserved_penalty_per_order
                self._step_delta.penalty_cost += pen
                self._cumulative.penalty_cost += pen
                self.event_log.append(
                    timestamp_minute=self.env.now,
                    event_type=EventType.EXCEPTION_RAISED,
                    entity_id=rec.order.order_id,
                    entity_type="unserved_at_horizon",
                    order_id=rec.order.order_id,
                    payload={"status": str(rec.status)},
                )

    def get_all_orders_runtime(self) -> list[OrderRuntimeRecord]:
        """Expose completed runtime records for post-run KPI computation."""
        return list(self._all_orders.values())
