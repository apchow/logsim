"""Scenario overlay transformations and dynamic runtime disruption tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.domain.entities import NetworkDataset, Order, Scenario
from app.domain.enums import HandlingClass, OverlayType


@dataclass
class ActiveDisruptionState:
    """Runtime disruption modifiers active at the current simulation timestamp."""

    transit_time_multiplier: float = 1.0
    cross_border_multiplier: float = 1.0
    inspection_time_multiplier: float = 1.0
    broken_vehicles: set[str] = field(default_factory=set)
    absent_pickers: set[str] = field(default_factory=set)
    hub_capacity_factor: dict[str, float] = field(default_factory=dict)
    active_overlay_labels: list[str] = field(default_factory=list)
    sku_shock_active: bool = False


def apply_static_scenario_overlays(
    dataset: NetworkDataset,
    scenario: Scenario,
    seed: int,
) -> NetworkDataset:
    """Apply demand/mix/promise overlays deterministically prior to simulation start."""
    if not scenario.overlays:
        return dataset

    rng = np.random.default_rng(seed + 7919)
    orders = [o.model_copy(deep=True) for o in dataset.orders]
    sku_by_id = {s.sku_id: s for s in dataset.skus}
    hazmat_skus = [s for s in dataset.skus if s.handling_class == HandlingClass.HAZMAT_BATTERY]

    for overlay in scenario.overlays:
        end_min = overlay.end_minute if overlay.end_minute is not None else scenario.horizon_minutes
        if overlay.overlay_type == OverlayType.DEMAND_MULTIPLIER:
            factor = float(overlay.value)
            if factor < 1.0:
                kept: list[Order] = []
                for order in orders:
                    if overlay.start_minute <= order.created_minute <= end_min:
                        if rng.random() <= factor:
                            kept.append(order)
                    else:
                        kept.append(order)
                orders = kept
            elif factor > 1.0:
                extra: list[Order] = []
                in_window = [
                    o for o in orders if overlay.start_minute <= o.created_minute <= end_min
                ]
                extra_count = int(round(len(in_window) * (factor - 1.0)))
                for idx in range(extra_count):
                    if not in_window:
                        break
                    src = in_window[int(rng.integers(0, len(in_window)))]
                    jitter = float(rng.uniform(-25.0, 25.0))
                    created = float(
                        np.clip(src.created_minute + jitter, overlay.start_minute, end_min - 60.0)
                    )
                    dur = max(60.0, src.promise_window_end_minute - src.promise_window_start_minute)
                    p_start = round(created + 30.0, 2)
                    p_end = round(min(scenario.horizon_minutes, p_start + dur), 2)
                    extra.append(
                        src.model_copy(
                            update={
                                "order_id": f"{src.order_id}-AUG{idx+1:03d}",
                                "created_minute": round(created, 2),
                                "promise_window_start_minute": p_start,
                                "promise_window_end_minute": max(p_start + 30.0, p_end),
                                "cancel_minute": (
                                    round(created + 90.0, 2)
                                    if src.cancel_minute is not None
                                    else None
                                ),
                            }
                        )
                    )
                orders.extend(extra)

        elif overlay.overlay_type == OverlayType.DEMAND_MIX_SHIFT and hazmat_skus:
            target_share = float(overlay.params.get("hazmat_share", overlay.value))
            shifted: list[Order] = []
            for order in orders:
                if overlay.start_minute <= order.created_minute <= end_min:
                    sku = sku_by_id[order.sku_id]
                    if (
                        sku.handling_class != HandlingClass.HAZMAT_BATTERY
                        and rng.random() < target_share
                    ):
                        new_sku = hazmat_skus[int(rng.integers(0, len(hazmat_skus)))]
                        shifted.append(
                            order.model_copy(
                                update={
                                    "sku_id": new_sku.sku_id,
                                    "requires_safety_segregation": True,
                                }
                            )
                        )
                        continue
                shifted.append(order)
            orders = shifted

        elif overlay.overlay_type == OverlayType.NEW_CUSTOMER_PROMISE_POLICY:
            factor = max(0.3, float(overlay.value))
            tightened: list[Order] = []
            for order in orders:
                if overlay.start_minute <= order.created_minute <= end_min:
                    window = order.promise_window_end_minute - order.promise_window_start_minute
                    new_end = round(
                        max(
                            order.promise_window_start_minute + 90.0,
                            order.promise_window_start_minute + window * factor,
                        ),
                        2,
                    )
                    tightened.append(order.model_copy(update={"promise_window_end_minute": new_end}))
                else:
                    tightened.append(order)
            orders = tightened

    orders = [o for o in orders if o.created_minute < scenario.horizon_minutes - 30.0]
    orders.sort(key=lambda o: (o.created_minute, o.order_id))
    return dataset.model_copy(update={"orders": orders})
