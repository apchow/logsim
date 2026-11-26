"""Consolidated tests for aligned FastAPI routes, MCP call_tool, examples/demo support, CLI, and 8s AI timeout."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.api.main import app
from app.cli.main import app as cli_app
from app.evaluation.ai_analyst import analyze_experiment_with_gemini
from app.mcp_server.server import mcp
from app.mcp_server.tools import tool_optimize_routes, tool_validate_dataset

client = TestClient(app)
cli_runner = CliRunner()


def test_consolidated_fastapi_and_mcp_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    # 1. Create scenario via POST /scenarios
    r_scn = client.post(
        "/scenarios",
        json={
            "scenario_id": "consolidated_test_scn",
            "name": "Consolidated Test Scenario",
            "overlays": [{"overlay_type": "demand_multiplier", "start_minute": 0.0, "value": 1.15}],
        },
    )
    assert r_scn.status_code == 200, r_scn.text
    assert Path(r_scn.json()["path"]).exists()

    # 2. Generate synthetic snapshot & validate via POST /datasets/validate
    r_syn = client.post(
        "/datasets/synthetic",
        json={"seed": 7, "horizon_minutes": 1440.0, "orders_per_day": 40, "output_dir": "data/syn"},
    )
    assert r_syn.status_code == 200, r_syn.text
    r_val = client.post("/datasets/validate", json={"dataset_path": "data/syn"})
    assert r_val.status_code == 200, r_val.text
    assert r_val.json()["valid"] is True

    # Missing dataset should fail closed with 404
    r_missing = client.post("/datasets/validate", json={"dataset_path": "data/does_not_exist"})
    assert r_missing.status_code == 404

    # 3. Run /backtest then inspect /runs/{run_id}/events, /runs/{run_id}/explain, and /experiments/.../compare
    r_bt = client.post(
        "/backtest",
        json={
            "dataset_path": "data/syn",
            "scenario_ids": ["consolidated_test_scn"],
            "policy_ids": ["current_replay", "fifo_edf"],
            "seeds": [101, 202],
            "experiment_name": "api_consolidated_exp",
            "baseline_policy_id": "current_replay",
            "routing_provider": "osrm",
        },
    )
    assert r_bt.status_code == 200, r_bt.text
    bt_body = r_bt.json()
    assert bt_body["run_count"] == 4
    assert Path(bt_body["html_path"]).exists()

    runs = client.get("/runs", params={"experiment_name": "api_consolidated_exp"}).json()
    assert len(runs) == 4
    run_id = runs[0]["run_id"]

    r_evs = client.get(f"/runs/{run_id}/events", params={"limit": 50})
    assert r_evs.status_code == 200
    assert len(r_evs.json()) > 0

    r_exp = client.get(f"/runs/{run_id}/explain")
    assert r_exp.status_code == 200
    assert "attribution" in r_exp.json()

    assert client.get("/runs/nonexistent_run/explain").status_code == 404

    r_cmp = client.get("/experiments/api_consolidated_exp/compare")
    assert r_cmp.status_code == 200
    assert len(r_cmp.json()["comparison"]) > 0

    # 4. POST /optimize/routes + POST /routing/matrix
    r_opt = client.post(
        "/optimize/routes",
        json={
            "orders": [{"order_id": "RMA-99", "origin_id": "ZONE-NW", "destination_hub_id": "HUB-NORTH", "priority": 5}],
            "vehicles": [{"vehicle_id": "VEH-01", "home_hub_id": "HUB-NORTH", "available": True}],
        },
    )
    assert r_opt.status_code == 200
    assert r_opt.json()["assignments"][0]["action"] == "dispatch"

    r_mat = client.post(
        "/routing/matrix",
        json={
            "origins": [{"location_id": "HUB-NORTH", "x_km": 0.0, "y_km": 8.0}],
            "destinations": [{"location_id": "ZONE-NW", "x_km": -18.0, "y_km": 19.0}],
            "provider": "graphhopper",
        },
    )
    assert r_mat.status_code == 200
    assert r_mat.json()["distances_km"][0][0] > 0.0


def test_examples_demo_bundle_and_mcp_call_tool() -> None:
    demo_dir = Path("examples/demo")
    if demo_dir.exists():
        res_bundle = tool_validate_dataset(str(demo_dir))
        assert res_bundle["valid"] is True
        assert res_bundle["kind"] == "bundle_directory"
        assert res_bundle["order_count"] == 100
        assert len(res_bundle["verified_snapshots"]) >= 1

    # Dynamic dep-style depot/customer route optimization
    opt_dep = tool_optimize_routes(
        orders=[{"order_id": "SYN-1", "origin_id": "depot-0", "destination_id": "customer-0001", "priority": 3}],
        vehicles=[{"resource_id": "veh-0", "location_id": "depot-0", "available": True}],
    )
    assert opt_dep["assignments"][0]["vehicle_id"] == "veh-0"

    # Real asyncio.run(mcp.call_tool(...)) execution
    tool_res = asyncio.run(
        mcp.call_tool(
            "get_route_matrix",
            {
                "origins": [{"location_id": "O1", "x_km": 0.0, "y_km": 0.0}],
                "destinations": [{"location_id": "D1", "x_km": 6.0, "y_km": 8.0}],
                "provider": "planar",
            },
        )
    )
    assert tool_res is not None


def test_gemini_ai_analyst_timeout_falls_back_cleanly(monkeypatch) -> None:
    monkeypatch.setattr("app.evaluation.ai_analyst._load_api_key", lambda: "fake-key-for-timeout-test")

    class _SlowInteractions:
        def create(self, **kwargs):
            time.sleep(2.5)
            return SimpleNamespace(output_text="too late")

    class _FakeClient:
        def __init__(self, api_key: str, http_options=None) -> None:
            self.interactions = _SlowInteractions()

    import sys

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=SimpleNamespace(Client=_FakeClient)))

    df = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "policy_id": "current_replay",
                "seed": 101,
                "on_time_delivery_rate": 0.5,
                "fill_rate": 0.8,
                "total_cost": 1000.0,
                "cost_per_order": 10.0,
                "net_cost_after_recovery": 100.0,
                "resource_utilization": 0.5,
                "empty_miles_km": 50.0,
                "carbon_kg_co2": 15.0,
                "constraint_violations": 0,
            },
            {
                "scenario_id": "baseline",
                "policy_id": "ortools_optimizer",
                "seed": 101,
                "on_time_delivery_rate": 0.85,
                "fill_rate": 0.95,
                "total_cost": 820.0,
                "cost_per_order": 8.2,
                "net_cost_after_recovery": -90.0,
                "resource_utilization": 0.85,
                "empty_miles_km": 35.0,
                "carbon_kg_co2": 11.0,
                "constraint_violations": 0,
            },
        ]
    )

    t0 = time.perf_counter()
    res = analyze_experiment_with_gemini("timeout_exp", df, timeout_sec=0.15)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert res["source"] == "deterministic_fallback"
    assert "TimeoutError" in res["markdown"]


def test_cli_mcp_help_and_route_matrix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    r_mcp = cli_runner.invoke(cli_app, ["mcp", "--help"])
    assert r_mcp.exit_code == 0, r_mcp.output

    r_mat = cli_runner.invoke(
        cli_app,
        [
            "route-matrix",
            "--provider",
            "osrm",
            "--origins",
            '[{"location_id":"H1","x_km":0,"y_km":0}]',
            "--destinations",
            '[{"location_id":"Z1","x_km":3,"y_km":4}]',
            "--json",
        ],
    )
    assert r_mat.exit_code == 0, r_mat.output
    assert "matrix_sha256" in r_mat.output
