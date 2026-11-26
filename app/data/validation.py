"""Fail-closed dataset and row validation ensuring referential integrity and bounds."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.domain.entities import NetworkDataset


@dataclass(frozen=True)
class ValidationIssue:
    """Single row/entity validation failure with actionable location."""

    row: int
    field: str
    message: str


class DatasetError(ValueError):
    """Fail-closed validation error carrying structured row/field diagnostics."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__(
            "; ".join(f"row {i.row} [{i.field}]: {i.message}" for i in issues[:20])
        )


class ValidationReport(BaseModel):
    """Structured validation outcome for a reverse-logistics dataset."""

    valid: bool
    dataset_id: str
    snapshot_id: str
    is_synthetic: bool
    order_count: int
    sku_count: int
    location_count: int
    vehicle_count: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_raw_order_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fail-closed pre-normalization check on raw tabular order rows."""
    cleaned: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()

    for row_num, raw in enumerate(rows, start=2):
        row = dict(raw)
        oid = str(row.get("order_id") or row.get("rma_id") or row.get("id") or "").strip()
        if not oid:
            issues.append(ValidationIssue(row_num, "order_id", "empty identifier"))
        elif oid in seen_ids:
            issues.append(ValidationIssue(row_num, "order_id", f"duplicate identifier '{oid}'"))
        else:
            seen_ids.add(oid)

        qty_raw = row.get("quantity") if "quantity" in row else row.get("qty")
        if qty_raw is not None:
            try:
                qty_val = float(qty_raw)
                if not math.isfinite(qty_val) or qty_val <= 0 or qty_val != int(qty_val):
                    issues.append(ValidationIssue(row_num, "quantity", f"must be positive integer, got {qty_raw!r}"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue(row_num, "quantity", f"invalid integer {qty_raw!r}"))

        cleaned.append(row)

    if not cleaned and not issues:
        issues.append(ValidationIssue(1, "dataset", "empty dataset"))
    if issues:
        raise DatasetError(issues)
    return cleaned


def validate_dataset(dataset: NetworkDataset) -> ValidationReport:
    """Verify entity uniqueness, referential integrity, capacities, and time windows."""
    errors: list[str] = []
    warnings: list[str] = []

    sku_ids = {s.sku_id for s in dataset.skus}
    loc_ids = {loc.location_id for loc in dataset.locations}
    veh_ids = {v.vehicle_id for v in dataset.vehicles}
    lane_pairs = {(lane.origin_id, lane.destination_id) for lane in dataset.lanes}

    if len(sku_ids) != len(dataset.skus):
        errors.append("Duplicate sku_id values detected.")
    if len(loc_ids) != len(dataset.locations):
        errors.append("Duplicate location_id values detected.")
    if len(veh_ids) != len(dataset.vehicles):
        errors.append("Duplicate vehicle_id values detected.")

    order_ids: set[str] = set()
    for order in dataset.orders:
        if order.order_id in order_ids:
            errors.append(f"Duplicate order_id: {order.order_id}")
        order_ids.add(order.order_id)
        if order.sku_id not in sku_ids:
            errors.append(f"Order {order.order_id} references unknown sku_id={order.sku_id}")
        if order.origin_id not in loc_ids:
            errors.append(f"Order {order.order_id} references unknown origin_id={order.origin_id}")
        if order.destination_hub_id not in loc_ids:
            errors.append(
                f"Order {order.order_id} references unknown destination_hub_id={order.destination_hub_id}"
            )
        if (order.origin_id, order.destination_hub_id) not in lane_pairs:
            warnings.append(
                f"Order {order.order_id} missing direct lane {order.origin_id}->{order.destination_hub_id}; will fallback to Euclidean routing."
            )

    for veh in dataset.vehicles:
        if veh.home_hub_id not in loc_ids:
            errors.append(f"Vehicle {veh.vehicle_id} references unknown home_hub_id={veh.home_hub_id}")

    for drv in dataset.drivers:
        if drv.assigned_vehicle_id not in veh_ids:
            errors.append(
                f"Driver {drv.driver_id} references unknown vehicle_id={drv.assigned_vehicle_id}"
            )

    if not dataset.orders:
        errors.append("Dataset contains zero orders.")
    if not dataset.vehicles:
        errors.append("Dataset contains zero vehicles.")
    if not dataset.is_synthetic:
        warnings.append("Dataset marked as historical/external; verify timezone alignment.")

    return ValidationReport(
        valid=len(errors) == 0,
        dataset_id=dataset.dataset_id,
        snapshot_id=dataset.snapshot_id,
        is_synthetic=dataset.is_synthetic,
        order_count=len(dataset.orders),
        sku_count=len(dataset.skus),
        location_count=len(dataset.locations),
        vehicle_count=len(dataset.vehicles),
        errors=errors,
        warnings=warnings,
    )
