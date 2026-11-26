"""FastAPI service and Gradio UI smoke tests."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app
from app.dashboard.gradio_app import _load_experiment_view, build_demo
from app.evaluation.ai_analyst import analyze_experiment_with_gemini

client = TestClient(app)


def test_api_health_policies_and_scenarios() -> None:
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_pol = client.get("/policies")
    assert r_pol.status_code == 200
    assert "ortools_optimizer" in r_pol.json()["policies"]
    assert "ppo_agent" in r_pol.json()["policies"]

    r_scn = client.get("/scenarios")
    assert r_scn.status_code == 200
    assert len(r_scn.json()) >= 3


def test_gradio_demo_and_analyst_fallback(monkeypatch) -> None:
    monkeypatch.setattr("app.evaluation.ai_analyst._load_api_key", lambda: None)
    demo = build_demo()
    assert demo is not None

    df = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "policy_id": "current_replay",
                "seed": 101,
                "on_time_delivery_rate": 0.4,
                "fill_rate": 0.7,
                "total_cost": 1000.0,
                "cost_per_order": 10.0,
                "net_cost_after_recovery": 200.0,
                "resource_utilization": 0.5,
                "empty_miles_km": 100.0,
                "carbon_kg_co2": 20.0,
                "constraint_violations": 0,
            },
            {
                "scenario_id": "baseline",
                "policy_id": "ortools_optimizer",
                "seed": 101,
                "on_time_delivery_rate": 0.8,
                "fill_rate": 0.9,
                "total_cost": 800.0,
                "cost_per_order": 8.0,
                "net_cost_after_recovery": -100.0,
                "resource_utilization": 0.85,
                "empty_miles_km": 60.0,
                "carbon_kg_co2": 15.0,
                "constraint_violations": 0,
            },
        ]
    )
    res = analyze_experiment_with_gemini("test_exp", df, baseline_policy_id="current_replay")
    assert "markdown" in res and "ortools_optimizer" in res["markdown"]

    view = _load_experiment_view(None, None)
    assert len(view) == 13
