"""Typed tool functions backing both the internal MCP server and FastAPI endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from app.config import load_app_settings
from app.data.ingestion import (
    ingest_orders_file,
    read_tabular_rows_strict,
    verify_snapshot,
)
from app.data.normalization import load_dataset_snapshot
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_dataset as validate_network_dataset
from app.data.validation import validate_raw_order_rows
from app.domain.entities import (
    KPIResult,
    NetworkDataset,
    PublicOrderView,
    Scenario,
    ScenarioOverlay,
)
from app.domain.enums import HandlingClass, OrderStatus, OverlayType
from app.evaluation.attribution import compute_cost_and_bottleneck_attribution
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import run_backtest_suite, run_single_episode
from app.experiments.scenarios import load_scenarios
from app.policies.optimizers.ortools_dispatcher import ORToolsDispatchPolicy
from app.policies.registry import build_policy
from app.routing.adapters import GeoPoint, get_route_matrix
from app.simulation.engine import DecisionContext, HubCausalView, VehicleCausalView


def _df_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to strict JSON-safe dictionaries (replacing NaN with None)."""
    if df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return list(clean.to_dict(orient="records"))


def resolve_dataset_input(
    dataset_path: str | Path | None = None,
    seed: int = 42,
    routing_provider: str | None = None,
    routing_profile: str = "driving",
) -> NetworkDataset:
    """Resolve reverse-logistics snapshots, dep multi-table bundles, raw tabular files, or synthetic defaults."""
    if dataset_path is None or str(dataset_path).strip() == "":
        default_syn = Path("data/synthetic/default/dataset_snapshot.json")
        if routing_provider is None and default_syn.exists():
            return load_dataset_snapshot(default_syn)
        return generate_synthetic_dataset(
            seed=seed,
            routing_provider=routing_provider,
            routing_profile=routing_profile,
        )

    src = Path(dataset_path)
    if not src.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {src}")

    if src.is_dir() and (src / "dataset_snapshot.json").exists():
        return load_dataset_snapshot(src / "dataset_snapshot.json")

    if src.suffix.lower() == ".json":
        return load_dataset_snapshot(src)

    # Support directory bundles (e.g., examples/demo) and CSV/Parquet files without mutating source
    with tempfile.TemporaryDirectory(prefix="logisticsbt_ds_") as tmp_dir:
        ds, _ = ingest_orders_file(
            input_path=src,
            output_dir=Path(tmp_dir),
            dataset_id=src.stem or "dataset",
            is_synthetic=None,
        )
        return ds


def tool_simulate_once(
    scenario_id: str = "baseline_normal",
    policy_id: str = "ortools_optimizer",
    seed: int = 101,
    dataset_path: str | None = None,
    experiment_name: str = "mcp_simulate",
    routing_provider: str | None = None,
) -> dict[str, Any]:
    """Run one discrete-event simulation episode and record KPI & event logs to DuckDB/Parquet."""
    settings = load_app_settings()
    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        raise KeyError(f"Scenario '{scenario_id}' not found")

    dataset = resolve_dataset_input(
        dataset_path=dataset_path,
        seed=seed,
        routing_provider=routing_provider,
    )
    policy = build_policy(policy_id, ppo_model_path=settings.rl.model_output_path)
    registry = ExperimentRegistry(settings.duckdb_path)
    run = run_single_episode(
        dataset=dataset,
        scenario=scenarios[scenario_id],
        policy=policy,
        seed=seed,
        settings=settings,
        experiment_name=experiment_name,
        registry=registry,
        routing_provider=routing_provider,
    )
    return run.model_dump(mode="json")


def tool_run_backtest(
    scenario_ids: list[str],
    policy_ids: list[str],
    seeds: list[int] | None = None,
    experiment_name: str = "mcp_backtest",
    baseline_policy_id: str = "current_replay",
    dataset_path: str | None = None,
    routing_provider: str | None = None,
) -> dict[str, Any]:
    """Run a paired multi-seed discrete-event backtest and store artifacts in DuckDB/Parquet."""
    settings = load_app_settings()
    scenarios_map = load_scenarios()
    missing = [s for s in scenario_ids if s not in scenarios_map]
    if missing:
        raise KeyError(f"Scenario(s) not found: {missing}")
    selected = [scenarios_map[s] for s in scenario_ids]
    if not selected:
        raise ValueError(f"No scenario_ids provided: {scenario_ids}")

    use_seeds = seeds or settings.default_seeds
    dataset = resolve_dataset_input(
        dataset_path=dataset_path,
        seed=42,
        routing_provider=routing_provider,
    )
    out = run_backtest_suite(
        dataset=dataset,
        scenarios=selected,
        policy_ids=policy_ids,
        seeds=use_seeds,
        settings=settings,
        experiment_name=experiment_name,
        baseline_policy_id=baseline_policy_id,
        routing_provider=routing_provider,
    )
    return {
        "experiment_name": out["experiment_name"],
        "run_count": len(out["runs"]),
        "csv_path": str(out["csv_path"]),
        "parquet_path": str(out["parquet_path"]),
        "html_path": str(out["html_path"]),
        "pdf_path": str(out["pdf_path"]),
        "summary": _df_to_json_records(out["summary_df"]),
        "comparison": _df_to_json_records(out["comparison_df"]),
    }


def tool_get_backtest_results(
    experiment_name: str,
    policy_id: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Fetch KPI rows and per-scenario summary statistics from DuckDB."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    df = reg.get_kpi_dataframe(experiment_name=experiment_name)
    if policy_id and not df.empty:
        df = df[df["policy_id"] == policy_id]
    if scenario_id and not df.empty:
        df = df[df["scenario_id"] == scenario_id]
    summary_df = summarize_across_seeds(df) if not df.empty else pd.DataFrame()
    return {
        "row_count": len(df),
        "runs": _df_to_json_records(df),
        "summary": _df_to_json_records(summary_df),
    }


def tool_compare_policies(
    experiment_name: str,
    baseline_policy_id: str = "current_replay",
    metrics: list[str] | None = None,
    practical_threshold_pct: float = 3.0,
) -> dict[str, Any]:
    """Compare candidate policies against a baseline with paired tests, CIs, and effect sizes."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    df = reg.get_kpi_dataframe(experiment_name=experiment_name)
    if df.empty:
        return {"comparison": [], "note": f"No runs found for experiment '{experiment_name}'"}
    use_metrics = tuple(metrics) if metrics else (
        "total_cost",
        "on_time_delivery_rate",
        "resource_utilization",
        "empty_miles_km",
    )
    comp_df = compare_policies_paired(
        df,
        baseline_policy_id=baseline_policy_id,
        metrics=use_metrics,
        practical_threshold_pct=practical_threshold_pct,
    )
    return {
        "experiment_name": experiment_name,
        "baseline_policy_id": baseline_policy_id,
        "comparison": _df_to_json_records(comp_df),
    }


def tool_optimize_routes(
    orders: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    timestamp_minute: float = 600.0,
    solver_time_limit_sec: float = 1.0,
) -> dict[str, Any]:
    """Run constrained OR-Tools CP-SAT dispatch assignment over candidate orders and vehicles."""
    inferred_hub = "HUB-NORTH"
    if vehicles:
        first_v = vehicles[0]
        inferred_hub = str(first_v.get("home_hub_id", first_v.get("location_id", inferred_hub)))

    visible_orders: list[PublicOrderView] = []
    lane_dists: dict[tuple[str, str], float] = {}
    lane_mins: dict[tuple[str, str], float] = {}
    hub_ids_seen: set[str] = {inferred_hub}

    for o in orders[:10]:
        orig = str(o.get("origin_id", "ZONE-NW"))
        dest_raw = o.get("destination_hub_id", o.get("destination_id", o.get("dest_id")))
        hub = str(dest_raw) if dest_raw is not None else inferred_hub
        if orig == inferred_hub and dest_raw is not None:
            # Forward dep-style order originating at depot/hub toward customer destination
            route_stop = str(dest_raw)
            hub = orig
        else:
            route_stop = orig
        hub_ids_seen.add(hub)

        dist = float(o.get("estimated_distance_km", 15.0))
        lane_dists[(route_stop, hub)] = dist
        lane_mins[(route_stop, hub)] = max(5.0, (dist / 42.0) * 60.0)
        hc_raw = str(o.get("handling_class", "standard"))
        is_hazmat = bool(o.get("requires_safety_segregation", o.get("hazardous", False)))
        hc = HandlingClass.HAZMAT_BATTERY if is_hazmat and hc_raw == "standard" else HandlingClass(hc_raw)
        visible_orders.append(
            PublicOrderView(
                order_id=str(o["order_id"]),
                sku_id=str(o.get("sku_id", "SKU-APP-01")),
                category=str(o.get("category", "general")),
                quantity=int(o.get("quantity", 1)),
                total_weight_kg=float(o.get("total_weight_kg", 2.0)),
                total_volume_m3=float(o.get("total_volume_m3", 0.02)),
                handling_class=hc,
                requires_safety_segregation=(is_hazmat or hc == HandlingClass.HAZMAT_BATTERY),
                origin_id=route_stop,
                destination_hub_id=hub,
                created_minute=float(o.get("created_minute", 0.0)),
                promise_window_start_minute=float(o.get("promise_window_start_minute", 30.0)),
                promise_window_end_minute=float(o.get("promise_window_end_minute", timestamp_minute + 240.0)),
                reported_condition=str(o.get("reported_condition", "opened_good")),
                priority=int(o.get("priority", 2)),
                unit_value_usd=float(o.get("unit_value_usd", 120.0)),
                estimated_distance_km=dist,
                status=OrderStatus.PENDING,
            )
        )

    veh_views: list[VehicleCausalView] = []
    veh_ids: list[str] = []
    for v in sorted(vehicles, key=lambda x: str(x.get("vehicle_id", x.get("resource_id", "")))):
        vid = str(v.get("vehicle_id", v.get("resource_id")))
        veh_ids.append(vid)
        home_hub = str(v.get("home_hub_id", v.get("location_id", inferred_hub)))
        hub_ids_seen.add(home_hub)
        cur_loc = str(v.get("current_location_id", home_hub))
        for vo in visible_orders:
            lane_dists.setdefault((cur_loc, vo.origin_id), float(vo.estimated_distance_km))
            lane_mins.setdefault((cur_loc, vo.origin_id), max(5.0, (vo.estimated_distance_km / 42.0) * 60.0))
        veh_views.append(
            VehicleCausalView(
                vehicle_id=vid,
                home_hub_id=home_hub,
                current_location_id=cur_loc,
                available=bool(v.get("available", True)),
                broken_down=bool(v.get("broken_down", False)),
                hazmat_certified=bool(v.get("hazmat_certified", v.get("hazardous_certified", False))),
                capacity_kg=float(v.get("capacity_kg", 180.0)),
                capacity_m3=float(v.get("capacity_m3", 2.2)),
                capacity_units=int(v.get("capacity_units", 20)),
                remaining_shift_minutes=float(v.get("remaining_shift_minutes", 480.0)),
                busy_until_minute=float(v.get("busy_until_minute", 0.0)),
                cost_per_km=float(v.get("cost_per_km", 1.15)),
            )
        )

    hub_views = tuple(
        HubCausalView(
            hub_id=hid,
            total_docks=3,
            docks_in_use=0,
            total_inspection_bays=4,
            inspection_in_use=0,
            quarantine_units=0,
            restock_units=25,
            refurb_kit_units=35,
            storage_capacity_units=1200,
        )
        for hid in sorted(hub_ids_seen)
    )

    ctx = DecisionContext(
        timestamp_minute=timestamp_minute,
        epoch_index=0,
        horizon_minutes=4320.0,
        visible_orders=tuple(visible_orders),
        total_pending_backlog=len(visible_orders),
        vehicles=tuple(veh_views),
        hubs=hub_views,
        active_disruptions=(),
        lane_distances_km=lane_dists,
        lane_base_minutes=lane_mins,
    )

    solver = ORToolsDispatchPolicy(solver_time_limit_sec=solver_time_limit_sec)
    import numpy as np

    action = solver.select_action(
        np.zeros(1, dtype=np.float32),
        {"context": ctx, "vehicle_ids": veh_ids},
    )
    overflow_code = len(veh_ids) + 1
    assignments: list[dict[str, Any]] = []
    for idx, order in enumerate(visible_orders):
        code = int(action[idx])
        if code <= 0:
            assignments.append({"order_id": order.order_id, "action": "defer", "vehicle_id": None})
        elif code == overflow_code:
            assignments.append(
                {"order_id": order.order_id, "action": "overflow_carrier", "vehicle_id": None}
            )
        else:
            assignments.append(
                {
                    "order_id": order.order_id,
                    "action": "dispatch",
                    "vehicle_id": veh_ids[code - 1],
                }
            )
    return {
        "solver": "ortools_cp_sat",
        "solver_time_ms": round(solver.last_solve_ms, 3),
        "assignments": assignments,
    }


def tool_get_route_matrix(
    origins: list[dict[str, Any]],
    destinations: list[dict[str, Any]],
    profile: str = "driving",
    provider: str = "osrm",
    departure_minute: float = 0.0,
) -> dict[str, Any]:
    """Compute or replay a deterministic cached origin-destination distance & time matrix."""
    orig_pts = [GeoPoint.model_validate(o) for o in origins]
    dest_pts = [GeoPoint.model_validate(d) for d in destinations]
    matrix = get_route_matrix(
        origins=orig_pts,
        destinations=dest_pts,
        provider=provider,
        profile=profile,
        departure_minute=departure_minute,
    )
    return matrix.model_dump(mode="json")


def _load_run_events_df(
    reg: ExperimentRegistry,
    run_id: str,
    limit: int = 1000,
    matched_run: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load run events from DuckDB, falling back to archived events_parquet if needed."""
    df = reg.get_run_events(run_id=run_id, limit=limit)
    if not df.empty:
        return df

    meta = matched_run
    if meta is None:
        runs = reg.list_runs(limit=500)
        matched = [r for r in runs if r.get("run_id") == run_id]
        meta = matched[0] if matched else None
    if meta and "artifact_paths_json" in meta:
        try:
            paths = json.loads(str(meta["artifact_paths_json"]))
            pq_raw = paths.get("events_parquet")
            if pq_raw and Path(str(pq_raw)).exists():
                import polars as pl

                return pl.read_parquet(str(pq_raw)).head(limit).to_pandas()
        except Exception:
            pass
    return df


def tool_query_events(
    run_id: str,
    event_types: list[str] | None = None,
    min_minute: float | None = None,
    max_minute: float | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Query time-ordered immutable simulation events for a run."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    df = _load_run_events_df(reg, run_id=run_id, limit=max(limit, 1000))
    if not df.empty:
        if event_types:
            df = df[df["event_type"].isin(event_types)]
        if min_minute is not None:
            df = df[df["timestamp_minute"] >= min_minute]
        if max_minute is not None:
            df = df[df["timestamp_minute"] <= max_minute]
        df = df.head(limit)
    return {"run_id": run_id, "count": len(df), "events": _df_to_json_records(df)}


def tool_explain_decision(run_id: str, decision_id: str | None = None) -> dict[str, Any]:
    """Explain a run's policy decisions, causal observation_hash, and cost/bottleneck attribution."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    runs = reg.list_runs(limit=500)
    matched = [r for r in runs if r.get("run_id") == run_id]
    if not matched:
        raise KeyError(f"Run '{run_id}' not found in registry")
    attribution: dict[str, Any] = {}
    if "metrics_json" in matched[0]:
        kpi_obj = KPIResult.model_validate_json(str(matched[0]["metrics_json"]))
        attribution = compute_cost_and_bottleneck_attribution(kpi_obj)

    events_df = _load_run_events_df(reg, run_id=run_id, limit=2000, matched_run=matched[0])
    dec_df = (
        events_df[events_df["event_type"] == "policy_decision"]
        if not events_df.empty
        else events_df
    )
    if decision_id and not dec_df.empty:
        dec_df = dec_df[dec_df["entity_id"] == decision_id]
    decisions = []
    for _, row in dec_df.head(20).iterrows():
        decisions.append(
            {
                "decision_id": row["entity_id"],
                "timestamp_minute": row["timestamp_minute"],
                "payload": json.loads(str(row["payload_json"])),
            }
        )
    return {
        "run_id": run_id,
        "policy_id": matched[0].get("policy_id"),
        "scenario_id": matched[0].get("scenario_id"),
        "attribution": attribution,
        "sample_decisions": decisions,
    }


def tool_create_scenario(
    scenario_id: str,
    name: str,
    description: str = "",
    overlays: list[dict[str, Any]] | None = None,
    horizon_minutes: float = 4320.0,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Create and persist a validated, versioned scenario YAML under configs/scenarios/."""
    parsed_overlays = [
        ScenarioOverlay(
            overlay_type=OverlayType(str(ov["overlay_type"])),
            start_minute=float(ov.get("start_minute", 0.0)),
            end_minute=float(ov["end_minute"]) if ov.get("end_minute") is not None else None,
            target_id=ov.get("target_id"),
            value=float(ov.get("value", 1.0)),
            params=dict(ov.get("params", {})),
        )
        for ov in (overlays or [])
    ]
    scn = Scenario(
        scenario_id=scenario_id,
        version=version,
        name=name,
        description=description,
        horizon_minutes=horizon_minutes,
        overlays=parsed_overlays,
        is_synthetic=True,
    )
    out_path = Path("configs/scenarios") / f"{scenario_id}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(scn.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    return {"scenario_id": scn.scenario_id, "version": scn.version, "path": str(out_path)}


def tool_validate_dataset(dataset_path: str = "data/synthetic/default") -> dict[str, Any]:
    """Validate reverse-logistics snapshots, SHA-256 manifests, dep directory bundles, or CSV/Parquet files."""
    target = Path(dataset_path)
    if not target.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {target}")

    if (
        target.is_dir()
        and (target / "manifest.json").exists()
        and not (target / "dataset_snapshot.json").exists()
    ):
        manifest = verify_snapshot(target)
        return {"kind": "snapshot_manifest", "valid": True, "manifest": manifest.model_dump(mode="json")}

    if target.is_file() and target.suffix.lower() in {".csv", ".parquet", ".pq"}:
        df = read_tabular_rows_strict(target)
        rows = validate_raw_order_rows(df.to_dict(orient="records"))
        return {"kind": "tabular_orders", "valid": True, "order_count": len(rows), "path": str(target)}

    if (
        target.is_dir()
        and not (target / "dataset_snapshot.json").exists()
        and ((target / "orders.parquet").exists() or (target / "orders.csv").exists())
    ):
        ds = resolve_dataset_input(dataset_path=target)
        report = validate_network_dataset(ds)
        verified_snapshots = []
        snap_root = target / "snapshots"
        if snap_root.is_dir():
            for snap_dir in sorted(snap_root.iterdir()):
                if snap_dir.is_dir() and (snap_dir / "manifest.json").exists():
                    verified_snapshots.append(verify_snapshot(snap_dir).model_dump(mode="json"))
        return {
            "kind": "bundle_directory",
            **report.model_dump(mode="json"),
            "verified_snapshots": verified_snapshots,
        }

    ds = load_dataset_snapshot(target)
    report = validate_network_dataset(ds)
    return {"kind": "network_dataset", **report.model_dump(mode="json")}
