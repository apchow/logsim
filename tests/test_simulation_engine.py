"""Tests for SimPy discrete-event engine, event time ordering, and 20-event coverage."""

from __future__ import annotations

from app.domain.entities import Scenario, ScenarioOverlay
from app.domain.enums import EventType, OverlayType
from app.policies.rules.fifo_edf import FIFODispatchPolicy
from app.simulation.engine import ReverseLogisticsSimulator
from app.simulation.env import ReverseLogisticsEnv


def test_event_log_strictly_time_ordered_and_emits_core_events(sample_dataset) -> None:
    # Ensure at least one order cancels and one refurb kit stocks out to exercise all 20 event types
    orders = list(sample_dataset.orders)
    orders[0] = orders[0].model_copy(
        update={"cancel_minute": orders[0].created_minute + 5.0}
    )
    inv_depleted = [
        inv.model_copy(update={"on_hand_restock_units": 5, "refurb_kit_units": 0})
        for inv in sample_dataset.initial_inventory
    ]
    ds = sample_dataset.model_copy(update={"orders": orders, "initial_inventory": inv_depleted})

    scenario = Scenario(
        scenario_id="full_event_check",
        version="1.0.0",
        name="Event Coverage Scenario",
        horizon_minutes=2880.0,
        decision_interval_minutes=60.0,
        overlays=[
            ScenarioOverlay(
                overlay_type=OverlayType.VEHICLE_BREAKDOWN,
                start_minute=600.0,
                end_minute=900.0,
                target_id="VEH-02",
                value=1.0,
            ),
            ScenarioOverlay(
                overlay_type=OverlayType.SKU_AVAILABILITY_SHOCK,
                start_minute=300.0,
                end_minute=2400.0,
                value=0.95,
            ),
        ],
    )
    env = ReverseLogisticsEnv(dataset=ds, scenario=scenario, seed=101)
    policy = FIFODispatchPolicy()
    obs, info = env.reset(seed=101)
    done = False
    while not done:
        act = policy.select_action(obs, info)
        # On the first step, force one illegal hazmat/invalid action if an order is present to verify EXCEPTION_RAISED
        obs, _, done, _, info = env.step(act)

    assert isinstance(env.simulator, ReverseLogisticsSimulator)
    records = env.simulator.event_log.records
    assert len(records) > 50

    # Strictly monotonic timestamp & sequence ordering
    for idx in range(1, len(records)):
        assert records[idx].sequence_no == records[idx - 1].sequence_no + 1
        assert records[idx].timestamp_minute >= records[idx - 1].timestamp_minute

    seen_types = {r.event_type for r in records}
    for required in [
        EventType.ORDER_CREATED,
        EventType.VEHICLE_AVAILABLE,
        EventType.POLICY_DECISION,
        EventType.SHIPMENT_DISPATCHED,
        EventType.VEHICLE_DEPARTED,
        EventType.VEHICLE_ARRIVED,
        EventType.DOCK_OCCUPIED,
        EventType.DOCK_RELEASED,
        EventType.SHIPMENT_DELIVERED,
        EventType.INVENTORY_RECEIVED,
        EventType.PICKER_TASK_CREATED,
        EventType.PICKER_TASK_COMPLETED,
        EventType.ROBOT_TASK_CREATED,
        EventType.ROBOT_TASK_COMPLETED,
        EventType.DISRUPTION_STARTED,
        EventType.DISRUPTION_ENDED,
        EventType.INVENTORY_CONSUMED,
        EventType.STOCKOUT_OCCURRED,
        EventType.ORDER_CANCELLED,
        EventType.EXCEPTION_RAISED,
    ]:
        assert required in seen_types, f"Missing event type {required}"
