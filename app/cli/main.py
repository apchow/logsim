"""Typer CLI entrypoint (`logisticsbt`) for ingestion, validation, simulation, training, backtesting, reporting, exporting, routing, and MCP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from app.config import RLSettings, load_app_settings
from app.data.ingestion import ingest_orders_file, read_tabular_rows_strict, verify_snapshot
from app.data.normalization import load_dataset_snapshot, save_dataset_snapshot
from app.data.synthetic import generate_synthetic_dataset
from app.data.validation import validate_dataset, validate_raw_order_rows
from app.domain.entities import NetworkDataset
from app.domain.enums import LocationType

if TYPE_CHECKING:
    from app.routing.adapters import GeoPoint
from app.evaluation.reporting import (
    export_results_csv,
    export_results_parquet,
    generate_html_report,
    generate_pdf_report,
)
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import run_backtest_suite, run_single_episode
from app.experiments.scenarios import load_scenarios
from app.policies.registry import available_policy_ids, build_policy

app = typer.Typer(
    name="logisticsbt",
    help="Logistics Backtester CLI — Discrete-event backtesting for reverse-logistics policies.",
    no_args_is_help=True,
)
console = Console()


def _resolve_dataset(dataset_path: Path | None, seed: int = 42) -> NetworkDataset:
    if dataset_path is not None and dataset_path.exists():
        return load_dataset_snapshot(dataset_path)
    default_syn = Path("data/synthetic/default/dataset_snapshot.json")
    if default_syn.exists():
        return load_dataset_snapshot(default_syn)
    return generate_synthetic_dataset(seed=seed)


@app.command()
def ingest(
    input_path: Annotated[Path | None, typer.Option("--input", "-i", help="CSV/Parquet/JSON orders file")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Snapshot output directory")] = Path("data/synthetic/default"),
    synthetic: Annotated[bool, typer.Option("--synthetic/--historical", help="Generate synthetic dataset")] = False,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for synthetic generation")] = 42,
    orders_per_day: Annotated[int, typer.Option("--orders-per-day", help="Synthetic daily order rate")] = 110,
    dataset_id: Annotated[str | None, typer.Option("--dataset-id", help="Logical dataset identifier")] = None,
) -> None:
    """Ingest CSV/Parquet dataset or generate a reproducible synthetic reverse-logistics snapshot."""
    if synthetic or input_path is None:
        ds = generate_synthetic_dataset(
            seed=seed,
            orders_per_day=orders_per_day,
            dataset_id=dataset_id or "synthetic_reverse_v1",
        )
        out = save_dataset_snapshot(ds, output_dir)
        console.print(
            f"[green]Generated synthetic dataset[/green] id={ds.dataset_id} snapshot={ds.snapshot_id} "
            f"orders={len(ds.orders)} is_synthetic={ds.is_synthetic} -> {out}"
        )
        return

    ds, out = ingest_orders_file(
        input_path=input_path,
        output_dir=output_dir,
        dataset_id=dataset_id,
        is_synthetic=None,
    )
    console.print(
        f"[green]Ingested dataset[/green] id={ds.dataset_id} snapshot={ds.snapshot_id} "
        f"orders={len(ds.orders)} is_synthetic={ds.is_synthetic} -> {out}"
    )


@app.command(name="synthetic")
def synthetic_cmd(
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("data/synthetic/default"),
    seed: Annotated[int, typer.Option("--seed", min=0, help="Mandatory random seed")] = 42,
    orders_per_day: Annotated[int, typer.Option("--orders-per-day", min=10, help="Synthetic daily order rate")] = 110,
    horizon_minutes: Annotated[float, typer.Option("--horizon-minutes", help="Horizon in minutes")] = 4320.0,
) -> None:
    """Generate an explicitly marked synthetic reverse-logistics dataset bundle."""
    ds = generate_synthetic_dataset(
        seed=seed,
        horizon_minutes=horizon_minutes,
        orders_per_day=orders_per_day,
    )
    out = save_dataset_snapshot(ds, output_dir)
    console.print(
        f"[green]Synthetic bundle[/green] id={ds.dataset_id} snapshot={ds.snapshot_id} "
        f"orders={len(ds.orders)} seed={seed} -> {out}"
    )


@app.command()
def validate(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to snapshot dir, JSON, or CSV/Parquet file")] = Path("data/synthetic/default"),
) -> None:
    """Validate dataset schema, snapshot SHA-256 manifest, or raw CSV/Parquet rows."""
    if dataset.is_dir() and (dataset / "manifest.json").exists():
        manifest = verify_snapshot(dataset)
        console.print_json(manifest.model_dump_json())
        return
    if dataset.is_file() and dataset.suffix.lower() in {".csv", ".parquet", ".pq"}:
        df = read_tabular_rows_strict(dataset)
        rows = validate_raw_order_rows(df.to_dict(orient="records"))
        console.print(f"[green]Valid order rows:[/green] {len(rows)} rows in {dataset}")
        return
    ds = load_dataset_snapshot(dataset)
    report = validate_dataset(ds)
    console.print_json(report.model_dump_json())
    if not report.valid:
        raise typer.Exit(code=1)


@app.command()
def simulate(
    policy: Annotated[str, typer.Option("--policy", "-p", help="Policy ID")] = "ortools_optimizer",
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="Scenario ID")] = "baseline_normal",
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 101,
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d", help="Dataset snapshot path")] = None,
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment name")] = "cli_simulate",
    routing_provider: Annotated[
        str | None, typer.Option("--routing-provider", help="Optional routing provider: osrm, graphhopper, planar")
    ] = None,
) -> None:
    """Run a single discrete-event simulation episode and record to the registry."""
    settings = load_app_settings()
    scenarios = load_scenarios()
    if scenario not in scenarios:
        console.print(f"[red]Unknown scenario:[/red] {scenario}")
        raise typer.Exit(code=1)

    ds = _resolve_dataset(dataset, seed=seed)
    pol = build_policy(policy, ppo_model_path=settings.rl.model_output_path)
    reg = ExperimentRegistry(settings.duckdb_path)
    run = run_single_episode(
        dataset=ds,
        scenario=scenarios[scenario],
        policy=pol,
        seed=seed,
        settings=settings,
        experiment_name=experiment,
        registry=reg,
        routing_provider=routing_provider,
    )
    m = run.metrics
    table = Table(title=f"Simulation Result: {policy} | {scenario} | seed={seed}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k in [
        "total_orders",
        "completed_orders",
        "on_time_delivery_rate",
        "fill_rate",
        "total_cost",
        "cost_per_order",
        "net_cost_after_recovery",
        "resource_utilization",
        "empty_miles_km",
        "carbon_kg_co2",
        "constraint_violations",
    ]:
        table.add_row(k, str(getattr(m, k)))
    console.print(table)
    console.print(f"Run ID: {run.run_id} | Events: {run.artifact_paths['events_parquet']}")


@app.command()
def train(
    timesteps: Annotated[int, typer.Option("--timesteps", "-t", help="Total PPO training timesteps")] = 15000,
    seed: Annotated[int, typer.Option("--seed", help="Training seed")] = 42,
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d", help="Dataset snapshot path")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Model zip output path")] = None,
) -> None:
    """Train a Stable-Baselines3 PPO policy across reverse-logistics scenarios."""
    from app.policies.rl.ppo_trainer import train_ppo_policy

    settings = load_app_settings()
    scenarios_map = load_scenarios()
    train_scenarios = [
        scenarios_map[k]
        for k in ["baseline_normal", "peak_surge_breakdown", "storm_tight_sla"]
        if k in scenarios_map
    ] or list(scenarios_map.values())
    ds = _resolve_dataset(dataset, seed=seed)
    rl_cfg: RLSettings = settings.rl.model_copy(update={"total_timesteps": timesteps})
    res = train_ppo_policy(
        dataset=ds,
        scenarios=train_scenarios,
        rl_settings=rl_cfg,
        sim_settings=settings.simulation,
        seed=seed,
        output_path=output or settings.rl.model_output_path,
    )
    console.print(f"[green]Trained PPO agent:[/green] {res}")


@app.command()
def backtest(
    policies: Annotated[
        str,
        typer.Option("--policies", "-p", help="Comma-separated policy IDs"),
    ] = "current_replay,fifo_edf,nearest_resource,fixed_batch,greedy_cost,ortools_optimizer,ppo_agent",
    scenarios: Annotated[
        str,
        typer.Option("--scenarios", "-s", help="Comma-separated scenario IDs"),
    ] = "baseline_normal,peak_surge_breakdown,storm_tight_sla",
    seeds: Annotated[
        str,
        typer.Option("--seeds", help="Comma-separated integer seeds (min 5 recommended)"),
    ] = "101,202,303,404,505",
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment name")] = "reverse_mvp_backtest",
    baseline: Annotated[str, typer.Option("--baseline", "-b", help="Baseline policy ID for uplift")] = "current_replay",
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d", help="Dataset snapshot path")] = None,
    auto_train_ppo: Annotated[bool, typer.Option("--auto-train-ppo/--no-auto-train-ppo", help="Train PPO automatically if model file missing")] = True,
    ppo_timesteps: Annotated[int, typer.Option("--ppo-timesteps", help="Timesteps if auto-training PPO")] = 12000,
    routing_provider: Annotated[
        str | None, typer.Option("--routing-provider", help="Optional routing provider: osrm, graphhopper, planar")
    ] = None,
) -> None:
    """Run paired multi-seed backtest across policies and scenarios, then generate reports."""
    settings = load_app_settings()
    scenarios_map = load_scenarios()
    policy_list = [p.strip() for p in policies.split(",") if p.strip()]
    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()]
    seed_list = [int(x.strip()) for x in seeds.split(",") if x.strip()]

    unknown_pol = [p for p in policy_list if p not in available_policy_ids()]
    if unknown_pol:
        console.print(f"[red]Unknown policies:[/red] {unknown_pol}")
        raise typer.Exit(code=1)
    unknown_scn = [s for s in scenario_list if s not in scenarios_map]
    if unknown_scn:
        console.print(f"[red]Unknown scenarios:[/red] {unknown_scn}")
        raise typer.Exit(code=1)

    ds = _resolve_dataset(dataset, seed=42)
    selected_scenarios = [scenarios_map[s] for s in scenario_list]

    ppo_path = Path(settings.rl.model_output_path)
    if "ppo_agent" in policy_list and not ppo_path.exists():
        if auto_train_ppo:
            from app.policies.rl.ppo_trainer import train_ppo_policy

            console.print("[yellow]PPO checkpoint missing — training compact MlpPolicy PPO...[/yellow]")
            rl_cfg = settings.rl.model_copy(update={"total_timesteps": ppo_timesteps})
            train_ppo_policy(
                dataset=ds,
                scenarios=selected_scenarios,
                rl_settings=rl_cfg,
                sim_settings=settings.simulation,
                seed=42,
                output_path=ppo_path,
            )
        else:
            console.print(f"[red]PPO model missing at {ppo_path}. Run `logisticsbt train` first.[/red]")
            raise typer.Exit(code=1)

    out = run_backtest_suite(
        dataset=ds,
        scenarios=selected_scenarios,
        policy_ids=policy_list,
        seeds=seed_list,
        settings=settings,
        experiment_name=experiment,
        baseline_policy_id=baseline,
        routing_provider=routing_provider,
    )
    console.print(
        f"[green]Completed {len(out['runs'])} runs[/green] for experiment '{experiment}'."
    )
    console.print(f"CSV:     {out['csv_path']}")
    console.print(f"Parquet: {out['parquet_path']}")
    console.print(f"HTML:    {out['html_path']}")
    console.print(f"PDF:     {out['pdf_path']}")


@app.command()
def report(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment name")] = "reverse_mvp_backtest",
    baseline: Annotated[str, typer.Option("--baseline", "-b", help="Baseline policy ID")] = "current_replay",
) -> None:
    """Generate HTML and PDF comparison reports from DuckDB registry results."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    df = reg.get_kpi_dataframe(experiment_name=experiment)
    if df.empty:
        console.print(f"[red]No runs found for experiment '{experiment}'.[/red]")
        raise typer.Exit(code=1)

    summary_df = summarize_across_seeds(df)
    comp_df = compare_policies_paired(df, baseline_policy_id=baseline)
    rep_dir = settings.artifacts_dir / "reports" / experiment
    html_p = generate_html_report(df, summary_df, comp_df, rep_dir / "report.html", baseline)
    pdf_p = generate_pdf_report(df, summary_df, comp_df, rep_dir / "report.pdf", baseline)

    table = Table(title=f"Paired Policy Uplift vs {baseline} ({experiment})")
    for col in ["scenario_id", "policy_id", "metric", "policy_mean", "baseline_mean", "uplift_pct", "p_value_ttest", "verdict"]:
        table.add_column(col)
    for _, r in comp_df.head(30).iterrows():
        table.add_row(*[str(r[c]) for c in ["scenario_id", "policy_id", "metric", "policy_mean", "baseline_mean", "uplift_pct", "p_value_ttest", "verdict"]])
    console.print(table)
    console.print(f"Generated HTML: {html_p}\nGenerated PDF:  {pdf_p}")


@app.command()
def export(
    experiment: Annotated[str, typer.Option("--experiment", "-e", help="Experiment name")] = "reverse_mvp_backtest",
    fmt: Annotated[str, typer.Option("--format", "-f", help="csv or parquet")] = "parquet",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Target output file")] = None,
) -> None:
    """Export experiment KPI result set to CSV or Parquet."""
    settings = load_app_settings()
    reg = ExperimentRegistry(settings.duckdb_path)
    df = reg.get_kpi_dataframe(experiment_name=experiment)
    if df.empty:
        console.print(f"[red]No runs found for experiment '{experiment}'.[/red]")
        raise typer.Exit(code=1)

    fmt_lower = fmt.lower()
    target = output or (settings.artifacts_dir / "results" / experiment / f"export.{fmt_lower}")
    if fmt_lower == "csv":
        out = export_results_csv(df, target)
    elif fmt_lower == "parquet":
        out = export_results_parquet(df, target)
    else:
        console.print(f"[red]Unsupported export format:[/red] {fmt}")
        raise typer.Exit(code=1)
    console.print(f"[green]Exported {len(df)} rows -> {out}[/green]")


def _parse_geopoints(
    raw: str | None,
    dataset: NetworkDataset | None,
    default_hubs_only: bool,
) -> list[GeoPoint]:
    """Parse GeoPoints from JSON list, comma-separated dataset location IDs, or dataset defaults."""
    from app.routing.adapters import GeoPoint

    if raw is not None and raw.strip():
        stripped = raw.strip()
        if stripped.startswith("["):
            parsed: Any = json.loads(stripped)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("JSON origins/destinations must be a non-empty list.")
            return [GeoPoint.model_validate(item) for item in parsed]

        if dataset is None:
            raise ValueError("Dataset is required when specifying location IDs.")
        loc_map = {loc.location_id: loc for loc in dataset.locations}
        points: list[GeoPoint] = []
        for token in [tok.strip() for tok in stripped.split(",") if tok.strip()]:
            if token not in loc_map:
                raise KeyError(f"Unknown location_id '{token}' in dataset '{dataset.dataset_id}'")
            loc = loc_map[token]
            points.append(GeoPoint(location_id=loc.location_id, x_km=loc.x_km, y_km=loc.y_km))
        if not points:
            raise ValueError("At least one location ID must be provided.")
        return points

    if dataset is None:
        raise ValueError("Dataset is required when origins/destinations are omitted.")
    locs = (
        [loc for loc in dataset.locations if loc.location_type == LocationType.RETURN_HUB]
        if default_hubs_only
        else [loc for loc in dataset.locations if loc.location_type != LocationType.RETURN_HUB]
    )
    if not locs:
        locs = list(dataset.locations[:2] if default_hubs_only else dataset.locations)
    return [GeoPoint(location_id=loc.location_id, x_km=loc.x_km, y_km=loc.y_km) for loc in locs]


@app.command(name="mcp")
def mcp_cmd() -> None:
    """Start the internal MCP 2.x stdio server (`app.mcp_server.server.main`)."""
    from app.mcp_server.server import main as mcp_main

    mcp_main()


@app.command(name="route-matrix")
def route_matrix_cmd(
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="Routing provider: osrm, graphhopper, or planar")
    ] = "osrm",
    profile: Annotated[str, typer.Option("--profile", help="Routing profile (driving or truck)")] = "driving",
    departure_minute: Annotated[
        float, typer.Option("--departure-minute", help="Departure minute for hourly cache bucketing")
    ] = 0.0,
    origins: Annotated[
        str | None,
        typer.Option("--origins", help="Comma-separated location IDs or JSON list of GeoPoint objects"),
    ] = None,
    destinations: Annotated[
        str | None,
        typer.Option("--destinations", help="Comma-separated location IDs or JSON list of GeoPoint objects"),
    ] = None,
    dataset: Annotated[Path | None, typer.Option("--dataset", "-d", help="Dataset snapshot path")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url", help="Optional live OSRM/GraphHopper URL")] = None,
    cache_path: Annotated[
        Path | None, typer.Option("--cache-path", help="Persistent route matrix cache JSON path")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print full RouteMatrix as JSON")] = False,
) -> None:
    """Compute or replay a cached OSRM / GraphHopper / Planar origin-destination route matrix."""
    from app.routing.adapters import get_route_matrix

    valid_providers = {"osrm", "graphhopper", "planar", "planar_cached"}
    if provider.lower() not in valid_providers:
        console.print(
            f"[red]Unsupported routing provider:[/red] {provider}. Choose from osrm, graphhopper, planar."
        )
        raise typer.Exit(code=1)

    settings = load_app_settings()
    needs_dataset = (
        origins is None
        or destinations is None
        or not origins.strip().startswith("[")
        or not destinations.strip().startswith("[")
    )
    ds = _resolve_dataset(dataset, seed=42) if needs_dataset else None

    try:
        origin_pts = _parse_geopoints(origins, ds, default_hubs_only=True)
        dest_pts = _parse_geopoints(destinations, ds, default_hubs_only=False)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        console.print(f"[red]Invalid route-matrix locations:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    resolved_cache = cache_path or (settings.artifacts_dir / "cache" / "route_cache.json")
    matrix = get_route_matrix(
        origins=origin_pts,
        destinations=dest_pts,
        provider=provider.lower(),
        profile=profile,
        departure_minute=departure_minute,
        base_url=base_url,
        cache_path=resolved_cache,
    )

    if json_output:
        console.print_json(matrix.model_dump_json())
        return

    table = Table(
        title=(
            f"Route Matrix ({matrix.provider} | {matrix.profile} | "
            f"cache_hit={matrix.cache_hit})"
        )
    )
    table.add_column("Origin")
    table.add_column("Destination")
    table.add_column("Distance (km)", justify="right")
    table.add_column("Duration (min)", justify="right")
    for i, orig_id in enumerate(matrix.origins):
        for j, dest_id in enumerate(matrix.destinations):
            table.add_row(
                orig_id,
                dest_id,
                f"{matrix.distances_km[i][j]:.3f}",
                f"{matrix.durations_minutes[i][j]:.2f}",
            )
    console.print(table)
    console.print(
        f"Provider={matrix.provider} | Profile={matrix.profile} | "
        f"CacheHit={matrix.cache_hit} | SHA256={matrix.matrix_sha256}"
    )


if __name__ == "__main__":
    app()
