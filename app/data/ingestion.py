"""CSV / Parquet / directory ingestion with atomic content-addressed snapshots."""

from __future__ import annotations

import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl

from app.data.normalization import (
    normalize_orders_frame,
    replace_orders_in_dataset,
    save_dataset_snapshot,
)
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_raw_order_rows
from app.domain.entities import (
    SKU,
    Driver,
    InventoryPosition,
    Lane,
    Location,
    NetworkDataset,
    Order,
    SnapshotManifest,
    Vehicle,
)
from app.domain.enums import LocationType


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_tabular_rows_strict(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet preserving string identifiers ('0001', 'NA') and rejecting duplicate headers."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = [c.strip() for c in next(reader)]
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate CSV columns detected in {path.name}")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        if len(set(df.columns)) != len(df.columns):
            raise ValueError(f"Duplicate normalized CSV columns detected in {path.name}")
        return df
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path).to_pandas()
    raise ValueError(f"Unsupported format '{suffix}'. Expected .csv, .parquet, or .json")


def _canonical_orders_bytes(orders: list[Order]) -> bytes:
    return ("\n".join(o.model_dump_json() for o in orders) + "\n").encode("utf-8")


def publish_content_addressed_snapshot(
    orders: list[Order],
    source_bytes: bytes,
    source_name: str,
    snapshots_root: Path,
    synthetic: bool,
) -> tuple[Path, SnapshotManifest]:
    """Atomically publish immutable content-addressed snapshot (orders.parquet + orders.jsonl + manifest.json)."""
    canonical = _canonical_orders_bytes(orders)
    canonical_sha = _sha256_bytes(canonical)
    source_sha = _sha256_bytes(source_bytes)
    snap_id = _sha256_bytes((canonical_sha + source_sha + source_name).encode("utf-8"))[:24]

    snapshots_root.mkdir(parents=True, exist_ok=True)
    target = snapshots_root / snap_id
    if target.exists():
        manifest = verify_snapshot(target)
        return target, manifest

    manifest = SnapshotManifest(
        snapshot_id=snap_id,
        synthetic=synthetic,
        source_sha256=source_sha,
        source_name=source_name,
        row_count=len(orders),
        canonical_sha256=canonical_sha,
    )
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshots_root))
    try:
        pl.DataFrame([o.model_dump(mode="python") for o in orders]).write_parquet(
            staging / "orders.parquet"
        )
        (staging / "orders.jsonl").write_bytes(canonical)
        (staging / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        try:
            staging.rename(target)
        except OSError:
            if not target.exists():
                raise
            verify_snapshot(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target, manifest


def verify_snapshot(snapshot_dir: str | Path) -> SnapshotManifest:
    """Verify SHA-256 canonical & byte-level integrity of a content-addressed snapshot."""
    target = Path(snapshot_dir)
    manifest = SnapshotManifest.model_validate_json(
        (target / "manifest.json").read_text(encoding="utf-8")
    )
    pq_df = pl.read_parquet(target / "orders.parquet")
    orders = [Order.model_validate(r) for r in pq_df.to_dicts()]
    canonical = _canonical_orders_bytes(orders)
    if _sha256_bytes(canonical) != manifest.canonical_sha256 or len(orders) != manifest.row_count:
        raise ValueError("Snapshot integrity check failed: canonical hash or row count mismatch")
    if (target / "orders.jsonl").read_bytes() != canonical:
        raise ValueError("Snapshot integrity check failed: orders.jsonl differs from orders.parquet")
    if {o.is_synthetic for o in orders} != {manifest.synthetic}:
        raise ValueError("Snapshot provenance mismatch between manifest and orders")
    return manifest


def _try_load_bundle_directory(bundle_dir: Path, base_network: NetworkDataset) -> NetworkDataset:
    """Adapt multi-file directory bundle (e.g., dep examples/demo) into NetworkDataset."""
    skus = base_network.skus
    locations = base_network.locations
    lanes = base_network.lanes
    vehicles = base_network.vehicles
    drivers = base_network.drivers
    calendars = base_network.calendars
    inventory = base_network.initial_inventory

    if (bundle_dir / "skus.parquet").exists():
        rows = pl.read_parquet(bundle_dir / "skus.parquet").to_dicts()
        if rows and "unit_weight_kg" in rows[0]:
            skus = [SKU.model_validate(r) for r in rows]
        elif rows:
            skus = [
                SKU(
                    sku_id=str(r["sku_id"]),
                    name=str(r.get("name", r["sku_id"])),
                    category=str(r.get("category", "general")),
                    unit_weight_kg=float(r.get("weight_kg", 1.5)),
                    unit_volume_m3=float(r.get("volume_m3", 0.02)),
                    unit_value_usd=float(r.get("unit_value_usd", 120.0)),
                    recovery_value_usd=float(r.get("recovery_value_usd", 80.0)),
                    inspection_minutes=float(r.get("inspection_minutes", 8.0)),
                )
                for r in rows
            ]

    if (bundle_dir / "locations.parquet").exists():
        rows = pl.read_parquet(bundle_dir / "locations.parquet").to_dicts()
        if rows and "location_type" in rows[0]:
            locations = [Location.model_validate(r) for r in rows]
        elif rows:
            locations = []
            for r in rows:
                kind = str(r.get("kind", r.get("type", "customer")))
                loc_type = (
                    LocationType.RETURN_HUB
                    if kind in {"depot", "hub", "warehouse"}
                    else LocationType.CUSTOMER_ZONE
                )
                dock_cap = int(r.get("dock_capacity", r.get("dock_count", 2 if loc_type == LocationType.RETURN_HUB else 0)))
                locations.append(
                    Location(
                        location_id=str(r["location_id"]),
                        name=str(r.get("name", r["location_id"])),
                        location_type=loc_type,
                        x_km=float(r.get("x_km", r.get("lon", 0.0))),
                        y_km=float(r.get("y_km", r.get("lat", 0.0))),
                        dock_count=max(2, dock_cap) if loc_type == LocationType.RETURN_HUB else 0,
                        inspection_bays=3 if loc_type == LocationType.RETURN_HUB else 0,
                        robot_units=2 if loc_type == LocationType.RETURN_HUB else 0,
                        storage_capacity_units=1200 if loc_type == LocationType.RETURN_HUB else 0,
                    )
                )

    if (bundle_dir / "lanes.parquet").exists():
        rows = pl.read_parquet(bundle_dir / "lanes.parquet").to_dicts()
        if rows and "base_transit_minutes" in rows[0]:
            lanes = [Lane.model_validate(r) for r in rows]
        elif rows:
            lanes = [
                Lane(
                    lane_id=str(r["lane_id"]),
                    origin_id=str(r["origin_id"]),
                    destination_id=str(r.get("destination_id", r.get("dest_id"))),
                    distance_km=float(r["distance_km"]),
                    base_transit_minutes=float(r.get("lead_time_minutes", r.get("base_duration_min", 15.0))),
                    cost_per_km=float(r.get("cost_per_km", 1.1)),
                )
                for r in rows
            ]

    if (bundle_dir / "resources.parquet").exists():
        rows = pl.read_parquet(bundle_dir / "resources.parquet").to_dicts()
        veh_rows = [r for r in rows if r.get("kind") == "vehicle"]
        if veh_rows:
            vehicles = [
                Vehicle(
                    vehicle_id=str(r["resource_id"]),
                    home_hub_id=str(r["location_id"]),
                    capacity_kg=max(10.0, float(r.get("capacity_kg", 150.0))),
                    capacity_m3=max(0.5, float(r.get("capacity_m3", 3.0))),
                    capacity_units=25,
                    hazmat_certified=bool(r.get("hazardous_certified", False)),
                    cost_per_km=float(r.get("cost_per_km", 1.0)) or 1.0,
                )
                for r in veh_rows
            ]
            drivers = [
                Driver(driver_id=f"DRV-{idx+1:02d}", assigned_vehicle_id=v.vehicle_id, max_shift_minutes=540.0)
                for idx, v in enumerate(vehicles)
            ]

    hub_locs = [loc for loc in locations if loc.location_type == LocationType.RETURN_HUB]
    if hub_locs and skus:
        inventory = [
            InventoryPosition(
                timestamp_minute=0.0,
                location_id=h.location_id,
                sku_id=s.sku_id,
                on_hand_restock_units=30,
                refurb_kit_units=35,
                reorder_point=10,
            )
            for h in hub_locs
            for s in skus
        ]

    return base_network.model_copy(
        update={
            "skus": skus,
            "locations": locations,
            "lanes": lanes,
            "vehicles": vehicles,
            "drivers": drivers,
            "calendars": calendars,
            "initial_inventory": inventory,
        }
    )


def ingest_orders_file(
    input_path: str | Path,
    output_dir: str | Path,
    dataset_id: str | None = None,
    base_network: NetworkDataset | None = None,
    is_synthetic: bool | None = None,
) -> tuple[NetworkDataset, Path]:
    """Ingest CSV/Parquet/JSON or directory bundle, validate fail-closed, publish atomic snapshot."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input path does not exist: {src}")

    if src.is_dir():
        if (src / "dataset_snapshot.json").exists():
            return _ingest_json_snapshot(src / "dataset_snapshot.json", Path(output_dir))
        orders_candidate = src / "orders.parquet"
        if not orders_candidate.exists():
            orders_candidate = src / "orders.csv"
        if not orders_candidate.exists():
            raise FileNotFoundError(f"Directory {src} has no dataset_snapshot.json, orders.parquet, or orders.csv")
        network = _try_load_bundle_directory(src, base_network or generate_synthetic_dataset(seed=42))
        src = orders_candidate
    else:
        network = base_network or generate_synthetic_dataset(seed=42)

    if src.suffix.lower() == ".json":
        return _ingest_json_snapshot(src, Path(output_dir))

    raw_pd = read_tabular_rows_strict(src)
    validate_raw_order_rows(raw_pd.to_dict(orient="records"))
    norm_df = normalize_orders_frame(raw_pd)

    syn_flags = set(norm_df["is_synthetic"].to_list())
    if len(syn_flags) > 1:
        raise ValueError("Mixed synthetic/historical provenance in dataset: split datasets first.")
    resolved_synthetic = (
        bool(next(iter(syn_flags)))
        if (is_synthetic is None and ("synthetic" in raw_pd.columns or "is_synthetic" in raw_pd.columns))
        else bool(is_synthetic)
    )

    resolved_id = dataset_id or src.stem
    dataset = replace_orders_in_dataset(
        base_dataset=network,
        normalized_orders_df=norm_df,
        dataset_id=resolved_id,
        is_synthetic=resolved_synthetic,
    )
    out = Path(output_dir)
    snapshot_path = save_dataset_snapshot(dataset, out)
    publish_content_addressed_snapshot(
        orders=dataset.orders,
        source_bytes=src.read_bytes(),
        source_name=src.name,
        snapshots_root=out / "snapshots",
        synthetic=resolved_synthetic,
    )
    return dataset, snapshot_path


def _ingest_json_snapshot(src: Path, output_dir: Path) -> tuple[NetworkDataset, Path]:
    dataset = NetworkDataset.model_validate_json(src.read_text(encoding="utf-8"))
    snapshot_path = save_dataset_snapshot(dataset, output_dir)
    return dataset, snapshot_path
