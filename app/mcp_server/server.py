"""Internal MCP server exposing typed Logistics Backtester tools over MCP 2.x."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

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

mcp = MCPServer("logistics-backtester")


@mcp.tool()
def run_backtest(
    scenario_ids: list[str],
    policy_ids: list[str],
    seeds: list[int] | None = None,
    experiment_name: str = "mcp_backtest",
    baseline_policy_id: str = "current_replay",
) -> dict[str, Any]:
    """Run a multi-seed paired backtest across scenarios and policies."""
    return tool_run_backtest(scenario_ids, policy_ids, seeds, experiment_name, baseline_policy_id)


@mcp.tool()
def get_backtest_results(
    experiment_name: str,
    policy_id: str | None = None,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve experiment KPIs and across-seed summaries from DuckDB."""
    return tool_get_backtest_results(experiment_name, policy_id, scenario_id)


@mcp.tool()
def compare_policies(
    experiment_name: str,
    baseline_policy_id: str = "current_replay",
    metrics: list[str] | None = None,
    practical_threshold_pct: float = 3.0,
) -> dict[str, Any]:
    """Perform paired policy vs. baseline statistical uplift analysis."""
    return tool_compare_policies(experiment_name, baseline_policy_id, metrics, practical_threshold_pct)


@mcp.tool()
def optimize_routes(
    orders: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    timestamp_minute: float = 600.0,
    solver_time_limit_sec: float = 0.25,
) -> dict[str, Any]:
    """Solve constrained pickup-dispatching assignment using Google OR-Tools CP-SAT."""
    return tool_optimize_routes(orders, vehicles, timestamp_minute, solver_time_limit_sec)


@mcp.tool()
def get_route_matrix(
    origins: list[dict[str, Any]],
    destinations: list[dict[str, Any]],
    profile: str = "driving",
    provider: str = "osrm",
    departure_minute: float = 0.0,
) -> dict[str, Any]:
    """Compute or replay cached OSRM / GraphHopper / planar distance and duration matrices."""
    return tool_get_route_matrix(origins, destinations, profile, provider, departure_minute)


@mcp.tool()
def query_events(
    run_id: str,
    event_types: list[str] | None = None,
    min_minute: float | None = None,
    max_minute: float | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Query immutable time-ordered discrete-event records for a simulation run."""
    return tool_query_events(run_id, event_types, min_minute, max_minute, limit)


@mcp.tool()
def explain_decision(run_id: str, decision_id: str | None = None) -> dict[str, Any]:
    """Inspect cost attribution, bottleneck classification, and causal observation_hash."""
    return tool_explain_decision(run_id, decision_id)


@mcp.tool()
def create_scenario(
    scenario_id: str,
    name: str,
    description: str = "",
    overlays: list[dict[str, Any]] | None = None,
    horizon_minutes: float = 4320.0,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Validate and save a versioned scenario overlay config under configs/scenarios/."""
    return tool_create_scenario(scenario_id, name, description, overlays, horizon_minutes, version)


@mcp.tool()
def validate_dataset(dataset_path: str = "data/synthetic/default") -> dict[str, Any]:
    """Validate a dataset snapshot directory or SHA-256 manifest."""
    return tool_validate_dataset(dataset_path)


def main() -> None:
    """Run the internal MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
