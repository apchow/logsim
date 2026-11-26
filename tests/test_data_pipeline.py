"""Tests for synthetic data generation, CSV/Parquet ingestion, normalization, and validation."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from app.data.ingestion import ingest_orders_file
from app.data.normalization import load_dataset_snapshot, save_dataset_snapshot
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_dataset


def test_synthetic_generator_deterministic_and_marked(tmp_path: Path) -> None:
    ds1 = generate_synthetic_dataset(seed=77, horizon_minutes=1440.0, orders_per_day=50)
    ds2 = generate_synthetic_dataset(seed=77, horizon_minutes=1440.0, orders_per_day=50)

    assert ds1.is_synthetic is True
    assert ds1.snapshot_id == ds2.snapshot_id
    assert [o.order_id for o in ds1.orders] == [o.order_id for o in ds2.orders]

    report = validate_dataset(ds1)
    assert report.valid is True
    assert report.errors == []

    snap = save_dataset_snapshot(ds1, tmp_path / "snap")
    loaded = load_dataset_snapshot(snap)
    assert loaded.snapshot_id == ds1.snapshot_id
    assert len(loaded.orders) == len(ds1.orders)


def test_csv_and_parquet_ingestion(tmp_path: Path) -> None:
    rows = [
        {
            "rma_id": "HIST-001",
            "sku_id": "SKU-APP-01",
            "qty": 2,
            "origin": "ZONE-NW",
            "hub_id": "HUB-NORTH",
            "created_at_min": 60.0,
            "due_minute": 420.0,
        },
        {
            "rma_id": "HIST-002",
            "sku_id": "SKU-ELC-01",
            "qty": 1,
            "origin": "ZONE-SE",
            "hub_id": "HUB-SOUTH",
            "created_at_min": 120.0,
            "due_minute": 540.0,
        },
    ]
    csv_path = tmp_path / "orders.csv"
    pq_path = tmp_path / "orders.parquet"
    pl.DataFrame(rows).write_csv(csv_path)
    pl.DataFrame(rows).write_parquet(pq_path)

    ds_csv, _ = ingest_orders_file(csv_path, tmp_path / "out_csv", dataset_id="hist_csv")
    ds_pq, _ = ingest_orders_file(pq_path, tmp_path / "out_pq", dataset_id="hist_pq")

    assert ds_csv.is_synthetic is False
    assert ds_pq.is_synthetic is False
    assert validate_dataset(ds_csv).valid is True
    assert validate_dataset(ds_pq).valid is True
    assert len(ds_csv.orders) == 2


def test_validator_catches_bad_references(sample_dataset) -> None:
    bad_order = sample_dataset.orders[0].model_copy(update={"sku_id": "UNKNOWN-SKU"})
    bad_ds = sample_dataset.model_copy(update={"orders": [bad_order, *sample_dataset.orders[1:]]})
    report = validate_dataset(bad_ds)
    assert report.valid is False
    assert any("UNKNOWN-SKU" in err for err in report.errors)
