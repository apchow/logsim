"""Cost and operational bottleneck attribution helpers."""

from __future__ import annotations

from typing import Any

from app.domain.entities import KPIResult


def compute_cost_and_bottleneck_attribution(kpi: KPIResult) -> dict[str, Any]:
    """Decompose total cost drivers and resource bottlenecks for a single run."""
    total = max(1e-6, kpi.total_cost)
    return {
        "run_id": kpi.run_id,
        "policy_id": kpi.policy_id,
        "scenario_id": kpi.scenario_id,
        "transport_share_pct": round(kpi.transport_cost / total * 100.0, 2),
        "processing_share_pct": round(kpi.processing_cost / total * 100.0, 2),
        "holding_share_pct": round(kpi.holding_cost / total * 100.0, 2),
        "penalty_share_pct": round(kpi.penalty_cost / total * 100.0, 2),
        "primary_bottleneck": (
            "dock_capacity"
            if kpi.dock_utilization > 0.80
            else "inspection_labor"
            if kpi.inspector_utilization > 0.80
            else "fleet_or_sla"
            if kpi.late_orders > 0
            else "balanced"
        ),
    }
