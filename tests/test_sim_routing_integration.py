"""Tests for cached routing adapter integration, 2-opt stop ordering, and simulator provenance."""

from __future__ import annotations

import json
from pathlib import Path

from app.data.synthetic import generate_synthetic_dataset
from app.domain.enums import EventType
from app.experiments.runner import run_backtest_suite
from app.policies.rules.fifo_edf import FIFODispatchPolicy
from app.routing.adapters import GeoPoint, get_route_matrix
from app.simulation.engine import OrderRuntimeRecord, ReverseLogisticsSimulator


def test_in_memory_route_matrix_without_disk_cache(tmp_path: Path) -> None:
    pts = [
        GeoPoint(location_id="A", x_km=0.0, y_km=0.0),
        GeoPoint(location_id="B", x_km=10.0, y_km=0.0),
    ]
    mat_truck = get_route_matrix(pts, pts, provider="planar", profile="truck", cache_path=None)
    mat_drive = get_route_matrix(pts, pts, provider="planar", profile="driving", cache_path=None)
    assert mat_truck.cache_hit is False
    assert mat_truck.distances_km[0][1] > mat_drive.distances_km[0][1]
    assert mat_truck.durations_minutes[0][1] > mat_drive.durations_minutes[0][1]
    assert not any(tmp_path.iterdir())


def test_synthetic_dataset_wires_routing_provider_deterministically(tmp_path: Path) -> None:
    cache_file = tmp_path / "syn_route_cache.json"
    ds1 = generate_synthetic_dataset(
        seed=42,
        horizon_minutes=1440.0,
        orders_per_day=40,
        routing_provider="osrm",
        routing_profile="truck",
        route_cache_path=cache_file,
    )
    ds2 = generate_synthetic_dataset(
        seed=42,
        horizon_minutes=1440.0,
        orders_per_day=40,
        routing_provider="osrm",
        routing_profile="truck",
        route_cache_path=cache_file,
    )
    assert ds1.snapshot_id == ds2.snapshot_id
    assert cache_file.exists()
    # Default call remains backward-compatible and distinct when routing profile overrides lanes
    ds_default = generate_synthetic_dataset(seed=42, horizon_minutes=1440.0, orders_per_day=40)
    assert ds1.lanes[0].emission_kg_co2_per_km == 0.34
    assert ds_default.lanes[0].emission_kg_co2_per_km == 0.27


def test_simulator_records_matrix_sha256_and_2opt_improves_route(
    sample_dataset, baseline_scenario, tmp_path: Path
) -> None:
    cache_file = tmp_path / "sim_cache.json"
    sim = ReverseLogisticsSimulator(
        dataset=sample_dataset,
        scenario=baseline_scenario,
        seed=101,
        routing_provider="graphhopper",
        routing_profile="driving",
        route_cache_path=cache_file,
    )
    assert len(sim.route_matrix_sha256) == 64
    assert sim.routing_provider == "graphhopper"

    # Verify _optimize_stop_sequence produces tour <= pure creation-order distance across hubs/zones
    sample_records = [
        OrderRuntimeRecord(order=o) for o in sample_dataset.orders[:5]
    ]
    hub_id = sim._resolve_hub_id(sample_records[0].order)
    optimized = sim._optimize_stop_sequence(hub_id, sample_records, hub_id)
    assert len(optimized) == len(sample_records)

    def _tour_dist(seq: list[OrderRuntimeRecord]) -> float:
        cur = hub_id
        tot = 0.0
        for rec in seq:
            loc = rec.order.origin_id if rec.order.origin_id != hub_id else rec.order.destination_hub_id
            tot += sim._lane_distance(cur, loc)
            cur = loc
        return tot + sim._lane_distance(cur, hub_id)

    assert _tour_dist(optimized) <= _tour_dist(sample_records) + 1e-6

    # Execute 1 decision step and confirm POLICY_DECISION records routing metadata
    ctx = sim.advance_to_first_decision()
    from app.domain.entities import OrderAssignment, PolicyDecision

    pol = FIFODispatchPolicy()
    # Run until at least one order becomes visible
    for _ in range(12):
        decision = PolicyDecision(
            timestamp_minute=ctx.timestamp_minute,
            epoch_index=ctx.epoch_index,
            policy_id=pol.policy_id,
            policy_type=pol.policy_type,
            policy_version=pol.policy_version,
            observation_hash="abc",
            assignments=[
                OrderAssignment(order_id=o.order_id, action_type="defer")
                for o in ctx.visible_orders
            ],
        )
        ctx, _, done = sim.apply_decision_and_step(decision)
        if done:
            break

    dec_events = [r for r in sim.event_log.records if r.event_type == EventType.POLICY_DECISION]
    assert dec_events
    assert dec_events[0].payload["matrix_sha256"] == sim.route_matrix_sha256
    assert dec_events[0].payload["routing_provider"] == "graphhopper"


def test_runner_records_routing_provenance_in_artifacts(
    sample_dataset, baseline_scenario, test_settings
) -> None:
    out = run_backtest_suite(
        dataset=sample_dataset,
        scenarios=[baseline_scenario],
        policy_ids=["current_replay", "fifo_edf"],
        seeds=[101],
        settings=test_settings,
        experiment_name="routing_prov_test",
        baseline_policy_id="current_replay",
        routing_provider="osrm",
    )
    assert len(out["runs"]) == 2
    for run in out["runs"]:
        assert run.artifact_paths["routing_provider"] == "osrm"
        assert len(run.artifact_paths["route_matrix_sha256"]) == 64
        # Artifact events parquet is readable
        assert Path(run.artifact_paths["events_parquet"]).exists()
        assert isinstance(json.loads(run.metrics.model_dump_json()), dict)
