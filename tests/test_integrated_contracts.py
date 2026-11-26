"""Integration tests covering Kimi_Agent_v and dep build artifacts, contracts, and snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from app.data.ingestion import ingest_orders_file, verify_snapshot
from app.domain.enums import EventType
from app.experiments.runner import run_single_episode
from app.experiments.scenarios import load_scenarios
from app.policies.rules.fifo_edf import FIFODispatchPolicy


def test_preserves_string_ids_and_rejects_duplicates_and_mixed_provenance(tmp_path: Path) -> None:
    # 1. String IDs "0001" and "NA" + ISO UTC timestamps
    valid_csv = tmp_path / "strict.csv"
    valid_csv.write_text(
        "order_id,sku_id,origin_id,destination_id,quantity,created_at,available_at,promise_at,synthetic\n"
        "0001,SKU-APP-01,ZONE-NW,HUB-NORTH,1,2026-01-01T10:00:00+05:30,2026-01-01T10:00:00+05:30,2026-01-01T14:00:00+05:30,false\n"
        "0002,SKU-APP-02,NA,HUB-SOUTH,2,2026-01-01T11:00:00+05:30,2026-01-01T11:00:00+05:30,2026-01-01T15:00:00+05:30,false\n",
        encoding="utf-8",
    )
    ds, snap_json = ingest_orders_file(valid_csv, tmp_path / "out_valid")
    assert ds.orders[0].order_id == "0001"
    assert ds.orders[1].origin_id == "NA"
    assert ds.is_synthetic is False

    # Verify atomic content-addressed snapshot integrity
    snap_dirs = list((tmp_path / "out_valid" / "snapshots").iterdir())
    assert len(snap_dirs) == 1
    manifest = verify_snapshot(snap_dirs[0])
    assert manifest.row_count == 2
    assert manifest.synthetic is False

    # Tamper detection
    (snap_dirs[0] / "orders.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        verify_snapshot(snap_dirs[0])

    # 2. Duplicate CSV headers rejected
    dup_csv = tmp_path / "dup.csv"
    dup_csv.write_text("order_id,order_id\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        ingest_orders_file(dup_csv, tmp_path / "out_dup")

    # 3. Mixed synthetic/historical provenance rejected
    mixed_pq = tmp_path / "mixed.parquet"
    pl.DataFrame(
        [
            {
                "order_id": "A1",
                "sku_id": "SKU-APP-01",
                "origin_id": "ZONE-NW",
                "destination_hub_id": "HUB-NORTH",
                "quantity": 1,
                "created_minute": 10.0,
                "promise_window_end_minute": 200.0,
                "synthetic": False,
            },
            {
                "order_id": "A2",
                "sku_id": "SKU-APP-01",
                "origin_id": "ZONE-NW",
                "destination_hub_id": "HUB-NORTH",
                "quantity": 1,
                "created_minute": 20.0,
                "promise_window_end_minute": 220.0,
                "synthetic": True,
            },
        ]
    ).write_parquet(mixed_pq)
    with pytest.raises(ValueError, match="Mixed synthetic"):
        ingest_orders_file(mixed_pq, tmp_path / "out_mixed")


def test_dep_demo_bundle_ingests_and_simulates(tmp_path: Path, test_settings) -> None:
    demo_dir = Path("examples/demo")
    if not demo_dir.exists():
        pytest.skip("examples/demo not present")
    ds, _ = ingest_orders_file(demo_dir, tmp_path / "demo_ingested", dataset_id="dep_demo")
    scenarios = load_scenarios()
    run = run_single_episode(
        dataset=ds,
        scenario=scenarios["baseline_normal"],
        policy=FIFODispatchPolicy(),
        seed=101,
        settings=test_settings,
        experiment_name="test_dep_demo",
    )
    assert run.status == "completed"
    assert run.metrics.total_orders == 100
    assert run.metrics.completed_orders > 0
    # Verify observation_hash present on POLICY_DECISION events
    events_df = pl.read_parquet(run.artifact_paths["events_parquet"])
    pol_rows = events_df.filter(pl.col("event_type") == str(EventType.POLICY_DECISION))
    assert pol_rows.height > 0
    payload0 = json.loads(pol_rows["payload_json"][0])
    assert len(payload0["observation_hash"]) == 64


def test_kimi_scenario_yamls_and_sql_ddl_load() -> None:
    scenarios = load_scenarios()
    for sid in ("baseline_normal", "baseline", "demand_surge", "carrier_delay"):
        assert sid in scenarios, f"Expected scenario '{sid}' to load"

    conn = duckdb.connect(":memory:")
    conn.execute(Path("docs/schema.sql").read_text(encoding="utf-8"))
    tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    assert {"experiment_runs", "kpi_results", "event_log"}.issubset(tables)
