"""KPI computation from completed SimPy reverse-logistics simulation runs."""

from __future__ import annotations

import numpy as np

from app.domain.entities import KPIResult
from app.domain.enums import EventType, OrderStatus
from app.simulation.engine import ReverseLogisticsSimulator


def compute_kpis(
    sim: ReverseLogisticsSimulator,
    policy_id: str,
    policy_version: str = "1.0.0",
) -> KPIResult:
    """Compute complete operational, financial, utilization, emission, and constraint KPIs."""
    records = sim.get_all_orders_runtime()
    total_orders = len(records)
    cancelled_orders = sum(1 for r in records if r.status == OrderStatus.CANCELLED)
    eligible_orders = max(1, total_orders - cancelled_orders)

    completed_records = [r for r in records if r.status == OrderStatus.COMPLETED]
    completed_orders = len(completed_records)
    on_time_orders = sim._cumulative.on_time_orders
    late_orders = sim._cumulative.late_orders

    cycle_times = [
        (r.completed_minute - r.order.created_minute)
        for r in completed_records
        if r.completed_minute is not None
    ]
    avg_cycle = float(np.mean(cycle_times)) if cycle_times else 0.0
    p95_cycle = float(np.percentile(cycle_times, 95)) if cycle_times else 0.0

    cum = sim._cumulative
    total_cost = round(
        cum.transport_cost + cum.holding_cost + cum.processing_cost + cum.penalty_cost, 2
    )
    recovery = round(cum.recovery_value, 2)
    net_cost = round(total_cost - recovery, 2)
    cost_per_order = round(total_cost / max(1, completed_orders or eligible_orders), 2)

    horizon_days = max(1e-6, sim.settings.horizon_minutes / 1440.0)
    # Vehicle utilization over available shift minutes across horizon
    total_shift_cap_min = sum(
        v.max_shift_minutes * horizon_days for v in sim.vehicles.values()
    )
    total_veh_busy_min = sum(v.total_busy_minutes for v in sim.vehicles.values())
    veh_util = float(np.clip(total_veh_busy_min / max(1.0, total_shift_cap_min), 0.0, 1.0))

    # Dock & inspection utilization over active hub hours (17h/day)
    active_hub_min = 17.0 * 60.0 * horizon_days
    total_dock_cap_min = sum(h.dock_res.capacity * active_hub_min for h in sim.hubs.values())
    total_dock_busy_min = sum(h.dock_busy_minutes for h in sim.hubs.values())
    dock_util = float(np.clip(total_dock_busy_min / max(1.0, total_dock_cap_min), 0.0, 1.0))

    total_insp_cap_min = sum(h.inspection_res.capacity * active_hub_min for h in sim.hubs.values())
    total_insp_busy_min = sum(h.inspection_busy_minutes for h in sim.hubs.values())
    insp_util = float(np.clip(total_insp_busy_min / max(1.0, total_insp_cap_min), 0.0, 1.0))

    backorders = sum(
        inv.backorder_units for h in sim.hubs.values() for inv in h.inventory.values()
    )
    exceptions_raised = sum(
        1 for ev in sim.event_log.records if ev.event_type == EventType.EXCEPTION_RAISED
    )

    return KPIResult(
        run_id=sim.run_id,
        policy_id=policy_id,
        policy_version=policy_version,
        scenario_id=sim.scenario.scenario_id,
        scenario_version=sim.scenario.version,
        seed=sim.seed,
        total_orders=total_orders,
        completed_orders=completed_orders,
        on_time_orders=on_time_orders,
        late_orders=late_orders,
        cancelled_orders=cancelled_orders,
        on_time_delivery_rate=round(on_time_orders / eligible_orders, 4),
        fill_rate=round(completed_orders / eligible_orders, 4),
        cost_per_order=cost_per_order,
        total_cost=total_cost,
        transport_cost=round(cum.transport_cost, 2),
        holding_cost=round(cum.holding_cost, 2),
        processing_cost=round(cum.processing_cost, 2),
        penalty_cost=round(cum.penalty_cost, 2),
        recovery_value=recovery,
        net_cost_after_recovery=net_cost,
        resource_utilization=round(veh_util, 4),
        dock_utilization=round(dock_util, 4),
        inspector_utilization=round(insp_util, 4),
        throughput_orders_per_day=round(completed_orders / horizon_days, 2),
        avg_cycle_time_minutes=round(avg_cycle, 2),
        p95_cycle_time_minutes=round(p95_cycle, 2),
        stockouts=cum.stockouts,
        backorders=backorders,
        empty_miles_km=round(cum.empty_miles_km, 2),
        total_miles_km=round(cum.total_miles_km, 2),
        carbon_kg_co2=round(cum.carbon_kg_co2, 2),
        constraint_violations=cum.constraint_violations,
        exceptions_raised=exceptions_raised,
    )
