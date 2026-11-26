"""Core Pydantic v2 entity models for the reverse-logistics domain."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import (
    DispositionOutcome,
    HandlingClass,
    LocationType,
    OrderStatus,
    OverlayType,
    PolicyType,
)


class SKU(BaseModel):
    """Stock-keeping unit metadata including reverse recovery & safety specs."""

    sku_id: str
    name: str
    category: str
    unit_weight_kg: float = Field(gt=0)
    unit_volume_m3: float = Field(gt=0)
    unit_value_usd: float = Field(ge=0)
    recovery_value_usd: float = Field(ge=0)
    handling_class: HandlingClass = HandlingClass.STANDARD
    inspection_minutes: float = Field(gt=0)
    restockable: bool = True
    refurbishable: bool = True
    refurb_kit_sku_id: str | None = None


class Location(BaseModel):
    """Network location: customer pickup zone, drop-off point, return hub, or disposition node."""

    location_id: str
    name: str
    location_type: LocationType
    x_km: float
    y_km: float
    dock_count: int = Field(default=0, ge=0)
    inspection_bays: int = Field(default=0, ge=0)
    robot_units: int = Field(default=0, ge=0)
    storage_capacity_units: int = Field(default=0, ge=0)
    calendar_id: str = "cal_standard"


class Lane(BaseModel):
    """Directed transport lane connecting two locations."""

    lane_id: str
    origin_id: str
    destination_id: str
    distance_km: float = Field(ge=0)
    base_transit_minutes: float = Field(ge=0)
    cost_per_km: float = Field(ge=0)
    fixed_dispatch_cost: float = Field(ge=0, default=12.0)
    emission_kg_co2_per_km: float = Field(ge=0, default=0.27)
    is_cross_border: bool = False
    active: bool = True


class Order(BaseModel):
    """Customer return order / RMA shipment in the reverse-logistics network."""

    order_id: str
    sku_id: str
    quantity: int = Field(gt=0)
    origin_id: str
    destination_hub_id: str
    created_minute: float = Field(ge=0)
    promise_window_start_minute: float = Field(ge=0)
    promise_window_end_minute: float = Field(gt=0)
    return_reason: str = "size_or_fit"
    reported_condition: str = "opened_good"
    # Ground-truth fields used internally by simulator; stripped from PublicOrderView!
    true_disposition: DispositionOutcome = DispositionOutcome.RESTOCK
    priority: int = Field(default=1, ge=1, le=5)
    requires_safety_segregation: bool = False
    cancel_minute: float | None = Field(default=None, ge=0)
    is_synthetic: bool = True

    @model_validator(mode="after")
    def validate_windows(self) -> Order:
        """Ensure promise window end is not earlier than start."""
        if self.promise_window_end_minute < self.promise_window_start_minute:
            raise ValueError("promise_window_end_minute must be >= promise_window_start_minute")
        if self.cancel_minute is not None and self.cancel_minute < self.created_minute:
            raise ValueError("cancel_minute must be >= created_minute")
        return self


class PublicOrderView(BaseModel):
    """Leakage-free view of an active order visible to decision policies at timestamp t."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    sku_id: str
    category: str
    quantity: int
    total_weight_kg: float
    total_volume_m3: float
    handling_class: HandlingClass
    requires_safety_segregation: bool
    origin_id: str
    destination_hub_id: str
    created_minute: float
    promise_window_start_minute: float
    promise_window_end_minute: float
    reported_condition: str
    priority: int
    unit_value_usd: float
    estimated_distance_km: float
    status: OrderStatus


class InventoryPosition(BaseModel):
    """Inventory state at a hub for a given SKU."""

    timestamp_minute: float = Field(ge=0)
    location_id: str
    sku_id: str
    on_hand_restock_units: int = Field(default=0, ge=0)
    quarantined_units: int = Field(default=0, ge=0)
    in_inspection_units: int = Field(default=0, ge=0)
    refurb_kit_units: int = Field(default=40, ge=0)
    backorder_units: int = Field(default=0, ge=0)
    reorder_point: int = Field(default=10, ge=0)


class Vehicle(BaseModel):
    """Reverse pickup vehicle with capacity, speed, emissions, and hazmat certification."""

    vehicle_id: str
    home_hub_id: str
    capacity_kg: float = Field(gt=0)
    capacity_m3: float = Field(gt=0)
    capacity_units: int = Field(gt=0)
    hazmat_certified: bool = False
    speed_kmph: float = Field(gt=0, default=45.0)
    cost_per_hour: float = Field(ge=0, default=32.0)
    cost_per_km: float = Field(ge=0, default=1.15)
    emission_kg_co2_per_km: float = Field(ge=0, default=0.27)
    calendar_id: str = "cal_driver_day"


class Driver(BaseModel):
    """Driver paired with a vehicle and bound by Hours-of-Service (HOS) rules."""

    driver_id: str
    assigned_vehicle_id: str
    max_shift_minutes: float = Field(gt=0, default=480.0)
    calendar_id: str = "cal_driver_day"


class Picker(BaseModel):
    """Human warehouse return inspector / grader at a hub."""

    picker_id: str
    hub_id: str
    inspection_speed_multiplier: float = Field(gt=0, default=1.0)
    calendar_id: str = "cal_hub_shift"


class Robot(BaseModel):
    """Autonomous mobile / sortation robot moving graded returns into disposition bins."""

    robot_id: str
    hub_id: str
    max_payload_kg: float = Field(gt=0, default=50.0)
    move_minutes_per_task: float = Field(gt=0, default=2.5)


class ResourceCalendar(BaseModel):
    """Operating shift and working-day calendar."""

    calendar_id: str
    shift_start_hour: int = Field(ge=0, le=23, default=6)
    shift_end_hour: int = Field(ge=1, le=24, default=22)
    working_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    max_continuous_hours: float = Field(gt=0, default=8.0)

    def is_active(self, minute_of_sim: float) -> bool:
        """Return True if the simulation minute falls inside working hours and days."""
        day = int(minute_of_sim // 1440.0) % 7
        hour = (minute_of_sim % 1440.0) / 60.0
        if day not in self.working_days:
            return False
        return float(self.shift_start_hour) <= hour < float(self.shift_end_hour)


class CapacityConstraint(BaseModel):
    """Explicit operational constraint enforced by the simulator guard."""

    constraint_id: str
    scope: Literal["vehicle", "dock", "hub_storage", "inspection", "driver_hos", "safety", "sla"]
    target_id: str
    max_weight_kg: float | None = Field(default=None, gt=0)
    max_volume_m3: float | None = Field(default=None, gt=0)
    max_units: int | None = Field(default=None, gt=0)
    max_concurrent_slots: int | None = Field(default=None, gt=0)
    max_shift_minutes: float | None = Field(default=None, gt=0)
    min_on_time_rate: float | None = Field(default=None, ge=0, le=1)
    require_hazmat_cert: bool = False
    hard: bool = True


class OrderAssignment(BaseModel):
    """Single order dispatching decision emitted by a policy."""

    order_id: str
    vehicle_id: str | None = None
    action_type: Literal["dispatch", "defer", "overflow_carrier"]


class PolicyDecision(BaseModel):
    """Recorded decision epoch payload from a policy with causal observation SHA-256."""

    schema_version: Literal["1.0"] = "1.0"
    decision_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    timestamp_minute: float = Field(ge=0)
    epoch_index: int = Field(ge=0)
    policy_id: str
    policy_type: PolicyType | str
    policy_version: str
    observation_hash: str | None = None
    assignments: list[OrderAssignment] = Field(default_factory=list)
    feasible: bool = True
    violation_count: int = Field(default=0, ge=0)
    violation_reasons: list[str] = Field(default_factory=list)
    solver_time_ms: float = Field(default=0.0, ge=0)


class ScenarioOverlay(BaseModel):
    """Composable, versioned operational perturbation or policy change."""

    overlay_type: OverlayType
    start_minute: float = Field(default=0.0, ge=0)
    end_minute: float | None = Field(default=None, ge=0)
    target_id: str | None = None
    value: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    """Versioned, reproducible simulation scenario definition."""

    scenario_id: str
    version: str = "1.0.0"
    name: str
    description: str = ""
    horizon_minutes: float = Field(gt=0, default=4320.0)
    decision_interval_minutes: float = Field(gt=0, default=60.0)
    overlays: list[ScenarioOverlay] = Field(default_factory=list)
    is_synthetic: bool = True


class KPIResult(BaseModel):
    """Full KPI summary computed from a completed simulation run's immutable event log."""

    run_id: str
    policy_id: str
    policy_version: str
    scenario_id: str
    scenario_version: str
    seed: int
    total_orders: int = Field(ge=0)
    completed_orders: int = Field(ge=0)
    on_time_orders: int = Field(ge=0)
    late_orders: int = Field(ge=0)
    cancelled_orders: int = Field(ge=0)
    on_time_delivery_rate: float = Field(ge=0, le=1)
    fill_rate: float = Field(ge=0, le=1)
    cost_per_order: float = Field(ge=0)
    total_cost: float = Field(ge=0)
    transport_cost: float = Field(ge=0, default=0.0)
    holding_cost: float = Field(ge=0, default=0.0)
    processing_cost: float = Field(ge=0, default=0.0)
    penalty_cost: float = Field(ge=0, default=0.0)
    recovery_value: float = Field(ge=0)
    net_cost_after_recovery: float
    resource_utilization: float = Field(ge=0, le=1)
    dock_utilization: float = Field(ge=0, le=1)
    inspector_utilization: float = Field(ge=0, le=1)
    throughput_orders_per_day: float = Field(ge=0)
    avg_cycle_time_minutes: float = Field(ge=0)
    p95_cycle_time_minutes: float = Field(ge=0)
    stockouts: int = Field(ge=0)
    backorders: int = Field(ge=0)
    empty_miles_km: float = Field(ge=0)
    total_miles_km: float = Field(ge=0)
    carbon_kg_co2: float = Field(ge=0)
    constraint_violations: int = Field(ge=0)
    exceptions_raised: int = Field(ge=0)


class ExperimentRun(BaseModel):
    """Reproducible experiment run ledger record."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    experiment_name: str
    config_hash: str
    git_commit: str
    git_dirty: bool = False
    dependency_lock_hash: str | None = None
    data_snapshot_id: str
    seed: int
    policy_id: str
    policy_type: str
    policy_version: str
    scenario_id: str
    scenario_version: str
    status: Literal["completed", "failed"]
    started_at: datetime
    finished_at: datetime
    runtime_seconds: float = Field(ge=0)
    metrics: KPIResult
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    mlflow_run_id: str | None = None


class SnapshotManifest(BaseModel):
    """Content-addressed dataset snapshot provenance manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: str
    synthetic: bool
    source_sha256: str
    source_name: str
    row_count: int = Field(ge=0)
    canonical_sha256: str
    generator_version: str | None = None
    seed: int | None = None


class SyntheticConfig(BaseModel):
    """Explicitly seeded synthetic reverse-logistics generation configuration."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    seed: int = Field(ge=0)
    horizon_minutes: float = Field(default=4320.0, gt=0)
    orders_per_day: int = Field(default=110, ge=10, le=5000)
    dataset_id: str = "synthetic_reverse_v1"


class OrderLine(BaseModel):
    """Optional multi-line SKU quantity item compatible with directory ingestion."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    sku_id: str
    quantity: int = Field(gt=0)


class Shipment(BaseModel):
    """Dispatched shipment batch linking vehicle, hub, orders, and miles."""

    model_config = ConfigDict(frozen=True)

    shipment_id: str
    vehicle_id: str
    hub_id: str
    order_ids: tuple[str, ...]
    dispatched_minute: float = Field(ge=0)
    delivered_minute: float | None = Field(default=None, ge=0)
    distance_km: float = Field(default=0.0, ge=0)
    empty_km: float = Field(default=0.0, ge=0)
    status: Literal["open", "completed", "failed"] = "open"


class NetworkDataset(BaseModel):
    """Normalized reverse-logistics dataset snapshot ready for simulation."""

    dataset_id: str
    snapshot_id: str
    is_synthetic: bool = True
    generated_at: datetime
    skus: list[SKU]
    locations: list[Location]
    lanes: list[Lane]
    orders: list[Order]
    initial_inventory: list[InventoryPosition]
    vehicles: list[Vehicle]
    drivers: list[Driver]
    pickers: list[Picker]
    robots: list[Robot]
    calendars: list[ResourceCalendar]
    constraints: list[CapacityConstraint]
