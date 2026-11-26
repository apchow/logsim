"""Data normalization, UTC-to-sim-minute conversion, and snapshot persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from app.domain.entities import NetworkDataset, Order

ORDER_COLUMN_ALIASES: dict[str, str] = {
    "rma_id": "order_id",
    "id": "order_id",
    "order_no": "order_id",
    "qty": "quantity",
    "origin": "origin_id",
    "origin_depot_id": "origin_id",
    "depot_id": "origin_id",
    "hub_id": "destination_hub_id",
    "destination_id": "destination_hub_id",
    "customer_id": "destination_hub_id",
    "created_at_min": "created_minute",
    "due_minute": "promise_window_end_minute",
    "synthetic": "is_synthetic",
}


def _parse_bool(val: Any, default: bool = False) -> bool:
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value for synthetic flag: {val!r}")


def _to_utc_datetime(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            raise ValueError(f"Naive timestamp not allowed (UTC/tz-aware required): {val!r}")
        return val.astimezone(UTC)
    ts = pd.Timestamp(str(val))
    if ts.tzinfo is None:
        raise ValueError(f"Naive timestamp not allowed (UTC/tz-aware required): {val!r}")
    dt: datetime = ts.to_pydatetime()
    return dt.astimezone(UTC)


def normalize_orders_frame(df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Normalize raw orders tabular data (minute-based or ISO UTC timestamps) into canonical Polars schema."""
    pd_df = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()
    pd_df.columns = [str(c).strip() for c in pd_df.columns]

    rename_map: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in pd_df.columns}
    for alias, canonical in ORDER_COLUMN_ALIASES.items():
        if alias in lower_cols and canonical not in pd_df.columns and canonical not in rename_map.values():
            rename_map[lower_cols[alias]] = canonical
    if rename_map:
        pd_df = pd_df.rename(columns=rename_map)

    # Convert ISO / UTC timestamps (`created_at`, `promise_at`, `promise_date`) into sim minutes if minute cols absent
    if "created_minute" not in pd_df.columns and "created_at" in pd_df.columns:
        created_dts = [_to_utc_datetime(v) for v in pd_df["created_at"]]
        anchor = min(created_dts)
        pd_df["created_minute"] = [
            round((dt - anchor).total_seconds() / 60.0, 4) for dt in created_dts
        ]
        if "available_at" in pd_df.columns:
            avail_dts = [_to_utc_datetime(v) for v in pd_df["available_at"]]
            for c_dt, a_dt in zip(created_dts, avail_dts, strict=True):
                if a_dt < c_dt:
                    raise ValueError("available_at cannot precede created_at")
            pd_df["promise_window_start_minute"] = [
                round((dt - anchor).total_seconds() / 60.0, 4) for dt in avail_dts
            ]
        promise_col = "promise_at" if "promise_at" in pd_df.columns else (
            "promise_date" if "promise_date" in pd_df.columns else None
        )
        if promise_col and "promise_window_end_minute" not in pd_df.columns:
            prom_dts = [_to_utc_datetime(v) for v in pd_df[promise_col]]
            for a_dt, p_dt in zip(created_dts, prom_dts, strict=True):
                if p_dt < a_dt:
                    raise ValueError("promise timestamp cannot precede created_at")
            pd_df["promise_window_end_minute"] = [
                round((dt - anchor).total_seconds() / 60.0, 4) for dt in prom_dts
            ]

    if "order_id" in pd_df.columns:
        pd_df["order_id"] = pd_df["order_id"].astype(str)
    if "origin_id" in pd_df.columns:
        pd_df["origin_id"] = pd_df["origin_id"].astype(str)
    if "destination_hub_id" in pd_df.columns:
        pd_df["destination_hub_id"] = pd_df["destination_hub_id"].astype(str)
    if "sku_id" in pd_df.columns:
        pd_df["sku_id"] = pd_df["sku_id"].astype(str)
    if "quantity" in pd_df.columns:
        pd_df["quantity"] = pd_df["quantity"].astype(int)
    if "created_minute" in pd_df.columns:
        pd_df["created_minute"] = pd_df["created_minute"].astype(float)
    if "promise_window_end_minute" in pd_df.columns:
        pd_df["promise_window_end_minute"] = pd_df["promise_window_end_minute"].astype(float)

    if "promise_window_start_minute" not in pd_df.columns and "created_minute" in pd_df.columns:
        pd_df["promise_window_start_minute"] = pd_df["created_minute"] + 30.0
    if "priority" not in pd_df.columns:
        pd_df["priority"] = 2
    if "is_synthetic" not in pd_df.columns:
        pd_df["is_synthetic"] = False
    else:
        pd_df["is_synthetic"] = [_parse_bool(v) for v in pd_df["is_synthetic"]]

    # Drop extra raw columns not in Order model (e.g. schema_version, created_at, service_minutes)
    allowed_cols = set(Order.model_fields.keys())
    keep_cols = [c for c in pd_df.columns if c in allowed_cols]
    pl_df = pl.from_pandas(pd_df[keep_cols])
    return pl_df.sort(["created_minute", "order_id"])


def save_dataset_snapshot(dataset: NetworkDataset, output_dir: str | Path) -> Path:
    """Persist normalized NetworkDataset bundle (JSON metadata + Parquet orders/lanes/SKUs)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    orders_df = pl.DataFrame([o.model_dump(mode="json") for o in dataset.orders])
    orders_df.write_parquet(out / "orders.parquet")

    skus_df = pl.DataFrame([s.model_dump(mode="json") for s in dataset.skus])
    skus_df.write_parquet(out / "skus.parquet")

    lanes_df = pl.DataFrame([lane.model_dump(mode="json") for lane in dataset.lanes])
    lanes_df.write_parquet(out / "lanes.parquet")

    bundle_path = out / "dataset_snapshot.json"
    bundle_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return bundle_path


def load_dataset_snapshot(path: str | Path) -> NetworkDataset:
    """Load a NetworkDataset snapshot from a directory or dataset_snapshot.json file."""
    target = Path(path)
    if target.is_dir():
        target = target / "dataset_snapshot.json"
    return NetworkDataset.model_validate_json(target.read_text(encoding="utf-8"))


def replace_orders_in_dataset(
    base_dataset: NetworkDataset,
    normalized_orders_df: pl.DataFrame,
    dataset_id: str,
    is_synthetic: bool = False,
) -> NetworkDataset:
    """Build a new immutable NetworkDataset snapshot from normalized orders table."""
    records = normalized_orders_df.to_dicts()
    orders = [Order.model_validate({**row, "is_synthetic": is_synthetic}) for row in records]
    payload = json.dumps(
        {
            "dataset_id": dataset_id,
            "n_orders": len(orders),
            "order_ids": [o.order_id for o in orders[:20]],
            "is_synthetic": is_synthetic,
        },
        sort_keys=True,
    )
    snapshot_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return base_dataset.model_copy(
        update={
            "dataset_id": dataset_id,
            "snapshot_id": snapshot_id,
            "is_synthetic": is_synthetic,
            "generated_at": datetime.now(UTC),
            "orders": orders,
        }
    )
