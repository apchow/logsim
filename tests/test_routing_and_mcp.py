"""Tests for cached OSRM/GraphHopper/Planar routing adapters and all 9 internal MCP tools."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.mcp_server.server import mcp
from app.mcp_server.tools import (
    tool_compare_policies,
    tool_create_scenario,
    tool_explain_decision,
    tool_get_backtest_results,
    tool_get_route_matrix,
    tool_optimize_routes,
    tool_query_events,
    tool_run_backtest,
    tool_validate_dataset,
)
from app.routing.adapters import GeoPoint, get_route_matrix


def test_cached_routing_adapters_reproducible(tmp_path: Path) -> None:
    cache_file = tmp_path / "route_cache.json"
    origins = [
        GeoPoint(location_id="HUB-NORTH", x_km=0.0, y_km=8.0, lat=40.75, lon=-73.99),
    ]
    dests = [
        GeoPoint(location_id="ZONE-NW", x_km=-18.0, y_km=19.0, lat=40.82, lon=-74.12),
        GeoPoint(location_id="ZONE-CORE", x_km=-2.0, y_km=1.0, lat=40.71, lon=-74.01),
    ]

    for provider in ("planar", "osrm", "graphhopper"):
        m1 = get_route_matrix(origins, dests, provider=provider, cache_path=cache_file)
        m2 = get_route_matrix(origins, dests, provider=provider, cache_path=cache_file)
        assert m1.cache_hit is False
        assert m2.cache_hit is True
        assert m1.matrix_sha256 == m2.matrix_sha256
        assert len(m1.distances_km) == 1 and len(m1.distances_km[0]) == 2
        assert m1.distances_km[0][0] > 0.0


def test_mcp_server_registers_all_nine_tools() -> None:
    tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    expected = {
        "run_backtest",
        "get_backtest_results",
        "compare_policies",
        "optimize_routes",
        "get_route_matrix",
        "query_events",
        "explain_decision",
        "create_scenario",
        "validate_dataset",
    }
    assert expected.issubset(tool_names)


def test_mcp_tool_handlers_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    # 1. create_scenario + validate_dataset (synthetic fallback)
    scn = tool_create_scenario(
        scenario_id="mcp_surge",
        name="MCP Surge",
        overlays=[{"overlay_type": "demand_multiplier", "start_minute": 0.0, "value": 1.1}],
    )
    assert Path(scn["path"]).exists()

    # 2. run_backtest + get_backtest_results + compare_policies + query_events + explain_decision
    bt = tool_run_backtest(
        scenario_ids=["baseline_normal"],
        policy_ids=["current_replay", "fifo_edf"],
        seeds=[101, 202],
        experiment_name="mcp_test_exp",
        baseline_policy_id="current_replay",
    )
    assert bt["run_count"] == 4

    res = tool_get_backtest_results("mcp_test_exp")
    assert res["row_count"] == 4
    run_id = res["runs"][0]["run_id"]

    comp = tool_compare_policies("mcp_test_exp", baseline_policy_id="current_replay")
    assert len(comp["comparison"]) > 0

    evs = tool_query_events(run_id, limit=25)
    assert evs["count"] > 0

    exp_info = tool_explain_decision(run_id)
    assert "attribution" in exp_info

    # 3. optimize_routes + get_route_matrix + validate_dataset
    opt = tool_optimize_routes(
        orders=[{"order_id": "RMA-1", "origin_id": "ZONE-NW", "destination_hub_id": "HUB-NORTH", "priority": 4}],
        vehicles=[{"vehicle_id": "VEH-01", "home_hub_id": "HUB-NORTH", "available": True}],
    )
    assert opt["assignments"][0]["action"] == "dispatch"

    mat = tool_get_route_matrix(
        origins=[{"location_id": "A", "x_km": 0.0, "y_km": 0.0}],
        destinations=[{"location_id": "B", "x_km": 3.0, "y_km": 4.0}],
        provider="osrm",
    )
    assert mat["distances_km"][0][0] > 0.0

    # Generate snapshot on disk and validate via tool
    from app.data.normalization import save_dataset_snapshot
    from app.data.synthetic import generate_synthetic_dataset

    save_dataset_snapshot(generate_synthetic_dataset(seed=7, horizon_minutes=1440.0), "data/syn")
    val = tool_validate_dataset("data/syn")
    assert val["valid"] is True
