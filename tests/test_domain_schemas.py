"""Tests for core Pydantic v2 domain schemas and immutable event stream."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.entities import Order
from app.domain.enums import EventType
from app.domain.events import EventLogRecord


def test_all_twenty_event_types_present() -> None:
    expected = {
        "order_created",
        "order_cancelled",
        "inventory_received",
        "inventory_consumed",
        "vehicle_available",
        "vehicle_departed",
        "vehicle_arrived",
        "dock_occupied",
        "dock_released",
        "picker_task_created",
        "picker_task_completed",
        "robot_task_created",
        "robot_task_completed",
        "disruption_started",
        "disruption_ended",
        "policy_decision",
        "shipment_dispatched",
        "shipment_delivered",
        "stockout_occurred",
        "exception_raised",
    }
    actual = {e.value for e in EventType}
    assert actual == expected


def test_event_record_is_immutable() -> None:
    rec = EventLogRecord(
        sequence_no=0,
        timestamp_minute=15.0,
        event_type=EventType.ORDER_CREATED,
        entity_id="RMA-001",
        entity_type="order",
        run_id="run-1",
        scenario_id="scn-1",
        seed=42,
    )
    with pytest.raises(ValidationError):
        rec.timestamp_minute = 25.0  # type: ignore[misc]


def test_order_promise_window_validation() -> None:
    with pytest.raises(ValidationError):
        Order(
            order_id="RMA-BAD",
            sku_id="SKU-APP-01",
            quantity=1,
            origin_id="ZONE-NW",
            destination_hub_id="HUB-NORTH",
            created_minute=100.0,
            promise_window_start_minute=200.0,
            promise_window_end_minute=150.0,
        )
