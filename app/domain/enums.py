"""Domain enumerations for reverse logistics simulation and backtesting."""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """All 20 supported immutable discrete-event types."""

    ORDER_CREATED = "order_created"
    ORDER_CANCELLED = "order_cancelled"
    INVENTORY_RECEIVED = "inventory_received"
    INVENTORY_CONSUMED = "inventory_consumed"
    VEHICLE_AVAILABLE = "vehicle_available"
    VEHICLE_DEPARTED = "vehicle_departed"
    VEHICLE_ARRIVED = "vehicle_arrived"
    DOCK_OCCUPIED = "dock_occupied"
    DOCK_RELEASED = "dock_released"
    PICKER_TASK_CREATED = "picker_task_created"
    PICKER_TASK_COMPLETED = "picker_task_completed"
    ROBOT_TASK_CREATED = "robot_task_created"
    ROBOT_TASK_COMPLETED = "robot_task_completed"
    DISRUPTION_STARTED = "disruption_started"
    DISRUPTION_ENDED = "disruption_ended"
    POLICY_DECISION = "policy_decision"
    SHIPMENT_DISPATCHED = "shipment_dispatched"
    SHIPMENT_DELIVERED = "shipment_delivered"
    STOCKOUT_OCCURRED = "stockout_occurred"
    EXCEPTION_RAISED = "exception_raised"


class HandlingClass(StrEnum):
    """Physical and safety handling classification of returned SKUs."""

    STANDARD = "standard"
    FRAGILE = "fragile"
    HAZMAT_BATTERY = "hazmat_battery"


class LocationType(StrEnum):
    """Node types in the reverse & forward logistics network."""

    CUSTOMER_ZONE = "customer_zone"
    DROPOFF_POINT = "dropoff_point"
    RETURN_HUB = "return_hub"
    REFURB_CENTER = "refurb_center"
    LIQUIDATION_DEPOT = "liquidation_depot"
    RECYCLING_FACILITY = "recycling_facility"
    DEPOT = "depot"
    CUSTOMER = "customer"
    HUB = "hub"
    WAREHOUSE = "warehouse"
    PORT = "port"


class DispositionOutcome(StrEnum):
    """Triage disposition path determined during hub inspection."""

    RESTOCK = "restock"
    REFURBISH = "refurbish"
    LIQUIDATE = "liquidate"
    RECYCLE = "recycle"


class OrderStatus(StrEnum):
    """Lifecycle states of a reverse-logistics return order."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    AT_DOCK = "at_dock"
    IN_INSPECTION = "in_inspection"
    SORTING = "sorting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PolicyType(StrEnum):
    """Policy families supported by the backtester."""

    CURRENT_REPLAY = "current_replay"
    RULE_BASED = "rule_based"
    OPTIMIZER = "optimizer"
    ML = "ml"
    RL = "rl"


class OverlayType(StrEnum):
    """Supported versioned scenario overlay types."""

    DEMAND_MULTIPLIER = "demand_multiplier"
    DEMAND_MIX_SHIFT = "demand_mix_shift"
    CARRIER_DELAY = "carrier_delay"
    LEAD_TIME_INCREASE = "lead_time_increase"
    WAREHOUSE_CAPACITY_REDUCTION = "warehouse_capacity_reduction"
    LABOR_ABSENCE = "labor_absence"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    PORT_CUSTOMS_DELAY = "port_customs_delay"
    WEATHER_DISRUPTION = "weather_disruption"
    SKU_AVAILABILITY_SHOCK = "sku_availability_shock"
    NEW_CUSTOMER_PROMISE_POLICY = "new_customer_promise_policy"
