"""Fast, lazy-loaded Gradio dashboard with in-memory caching and on-demand Gemini API insights."""

from __future__ import annotations

from typing import Any

import gradio as gr
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.config import load_app_settings
from app.domain.entities import KPIResult
from app.evaluation.attribution import compute_cost_and_bottleneck_attribution
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds
from app.experiments.registry import ExperimentRegistry
from app.experiments.scenarios import load_scenarios
from app.policies.registry import available_policy_ids

API_BASE_URL = "http://127.0.0.1:8000"

# In-memory UI caches keyed by (experiment_name, baseline_policy, row_count)
_VIEW_CACHE: dict[tuple[str, str, int], tuple[Any, ...]] = {}
_RUN_EVENT_CACHE: dict[str, tuple[Any, ...]] = {}


def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, height=400)
    return fig


def _resolve_existing_exports(chosen_exp: str) -> tuple[str | None, str | None, str | None, str | None]:
    """Return pre-built CSV/Parquet/HTML/PDF paths without regenerating them on page load."""
    settings = load_app_settings()
    res_dir = settings.artifacts_dir / "results" / chosen_exp
    rep_dir = settings.artifacts_dir / "reports" / chosen_exp

    def _pick(*candidates: Any) -> str | None:
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    return (
        _pick(rep_dir / "kpi_results.csv", res_dir / "kpi_results.csv"),
        _pick(rep_dir / "kpi_results.parquet", res_dir / "kpi_results.parquet"),
        _pick(rep_dir / "report.html"),
        _pick(rep_dir / "report.pdf"),
    )


def _load_experiment_view(
    experiment_name: str | None,
    baseline_policy: str | None,
) -> tuple[Any, ...]:
    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    df = registry.get_kpi_dataframe()
    if df.empty:
        empty_df = pd.DataFrame()
        return (
            "### No backtest runs recorded yet. Run a backtest in Tab 4 or via `logisticsbt backtest`.",
            empty_df,
            _empty_fig("Cost vs. On-Time Delivery"),
            empty_df,
            _empty_fig("Mean Total Cost by Scenario & Policy"),
            empty_df,
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            None,
            None,
            None,
            None,
        )

    exp_options = sorted(df["experiment_name"].unique().tolist())
    chosen_exp = experiment_name if experiment_name in exp_options else exp_options[-1]
    exp_df = df[df["experiment_name"] == chosen_exp].copy()

    pol_options = sorted(exp_df["policy_id"].unique().tolist())
    base_choice = (
        baseline_policy
        if baseline_policy in pol_options
        else ("current_replay" if "current_replay" in pol_options else pol_options[0])
    )

    cache_key = (chosen_exp, base_choice, len(exp_df))
    if cache_key in _VIEW_CACHE:
        cached = _VIEW_CACHE[cache_key]
        csv_p, pq_p, html_p, pdf_p = _resolve_existing_exports(chosen_exp)
        return (
            cached[0],
            cached[1],
            cached[2],
            cached[3],
            cached[4],
            cached[5],
            gr.update(choices=exp_options, value=chosen_exp),
            gr.update(choices=pol_options, value=base_choice),
            gr.update(choices=cached[6], value=cached[6][0] if cached[6] else None),
            csv_p,
            pq_p,
            html_p,
            pdf_p,
        )

    mean_cols = [
        "on_time_delivery_rate",
        "fill_rate",
        "total_cost",
        "cost_per_order",
        "net_cost_after_recovery",
        "resource_utilization",
        "empty_miles_km",
        "carbon_kg_co2",
        "constraint_violations",
    ]
    agg_df = (
        exp_df.groupby("policy_id", as_index=False)[mean_cols]
        .mean()
        .round(3)
        .sort_values("total_cost")
    )
    summary_df = summarize_across_seeds(exp_df)
    comp_df = compare_policies_paired(exp_df, baseline_policy_id=base_choice)

    best_otd = float(agg_df["on_time_delivery_rate"].max())
    lowest_cost_row = agg_df.iloc[0]
    banner = (
        f"### Experiment `{chosen_exp}` | Runs: **{len(exp_df)}** | "
        f"Policies: **{exp_df['policy_id'].nunique()}** | "
        f"Scenarios: **{exp_df['scenario_id'].nunique()}** | "
        f"Best Mean OTD: **{best_otd:.1%}** | "
        f"Lowest Gross Cost: **`{lowest_cost_row['policy_id']}` (${lowest_cost_row['total_cost']:,.2f})**"
    )

    fig_tradeoff = px.scatter(
        agg_df,
        x="total_cost",
        y="on_time_delivery_rate",
        size="resource_utilization",
        color="policy_id",
        hover_data=["fill_rate", "net_cost_after_recovery", "constraint_violations"],
        title="Cost vs. On-Time Delivery Rate (Mean Across Scenarios & Seeds)",
        height=420,
    )

    fig_scen = px.bar(
        exp_df.groupby(["scenario_id", "policy_id"], as_index=False)["total_cost"].mean().round(2),
        x="scenario_id",
        y="total_cost",
        color="policy_id",
        barmode="group",
        title="Mean Total Cost by Scenario & Policy",
        height=420,
    )

    run_ids = exp_df["run_id"].tolist()
    _VIEW_CACHE[cache_key] = (
        banner,
        agg_df,
        fig_tradeoff,
        comp_df,
        fig_scen,
        summary_df,
        run_ids,
    )
    csv_p, pq_p, html_p, pdf_p = _resolve_existing_exports(chosen_exp)

    return (
        banner,
        agg_df,
        fig_tradeoff,
        comp_df,
        fig_scen,
        summary_df,
        gr.update(choices=exp_options, value=chosen_exp),
        gr.update(choices=pol_options, value=base_choice),
        gr.update(choices=run_ids, value=run_ids[0] if run_ids else None),
        csv_p,
        pq_p,
        html_p,
        pdf_p,
    )


def _inspect_run(experiment_name: str | None, run_id: str | None) -> tuple[Any, ...]:
    if not run_id:
        return {}, {}, _empty_fig("No Run Selected"), pd.DataFrame()
    if run_id in _RUN_EVENT_CACHE:
        return _RUN_EVENT_CACHE[run_id]

    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    df = registry.get_kpi_dataframe(experiment_name=experiment_name)
    matched_df = df[df["run_id"] == run_id]
    kpi_dict = matched_df.iloc[0].to_dict() if not matched_df.empty else {}

    runs_meta = registry.list_runs(experiment_name=experiment_name, limit=500)
    matched_meta = [r for r in runs_meta if r.get("run_id") == run_id]
    attribution: dict[str, Any] = {}
    if matched_meta and "metrics_json" in matched_meta[0]:
        kpi_obj = KPIResult.model_validate_json(str(matched_meta[0]["metrics_json"]))
        attribution = compute_cost_and_bottleneck_attribution(kpi_obj)

    events_df = registry.get_run_events(run_id, limit=500)
    if events_df.empty:
        res = (
            attribution,
            kpi_dict,
            _empty_fig("Events archived to Parquet on disk for non-primary seed"),
            pd.DataFrame(),
        )
        _RUN_EVENT_CACHE[run_id] = res
        return res

    ev_counts = events_df["event_type"].value_counts().reset_index()
    ev_counts.columns = ["event_type", "count"]
    ev_fig = px.bar(
        ev_counts,
        x="event_type",
        y="count",
        title=f"Immutable Event Stream Distribution ({run_id})",
        height=380,
    )
    res = (attribution, kpi_dict, ev_fig, events_df)
    _RUN_EVENT_CACHE[run_id] = res
    return res


def _generate_exports_on_demand(
    experiment_name: str | None, baseline_policy: str | None
) -> tuple[str | None, str | None, str | None, str | None, str]:
    if not experiment_name:
        return None, None, None, None, "Select an experiment first."
    from app.evaluation.reporting import (
        export_results_csv,
        export_results_parquet,
        generate_html_report,
        generate_pdf_report,
    )

    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    exp_df = registry.get_kpi_dataframe(experiment_name=experiment_name)
    if exp_df.empty:
        return None, None, None, None, f"No runs found for `{experiment_name}`."

    base_choice = baseline_policy or sorted(exp_df["policy_id"].unique().tolist())[0]
    summary_df = summarize_across_seeds(exp_df)
    comp_df = compare_policies_paired(exp_df, baseline_policy_id=base_choice)
    rep_dir = settings.artifacts_dir / "reports" / experiment_name
    csv_p = str(export_results_csv(exp_df, rep_dir / "kpi_results.csv"))
    pq_p = str(export_results_parquet(exp_df, rep_dir / "kpi_results.parquet"))
    html_p = str(
        generate_html_report(exp_df, summary_df, comp_df, rep_dir / "report.html", base_choice)
    )
    pdf_p = str(
        generate_pdf_report(exp_df, summary_df, comp_df, rep_dir / "report.pdf", base_choice)
    )
    return csv_p, pq_p, html_p, pdf_p, f"Generated fresh exports for `{experiment_name}`."


def _ask_ai_analyst(
    experiment_name: str | None,
    baseline_policy: str | None,
    question: str,
) -> str:
    """Invoke Gemini AI analyst via FastAPI when running, or directly in-process."""
    if not experiment_name:
        return "Select an experiment first."
    payload = {
        "experiment_name": experiment_name,
        "baseline_policy_id": baseline_policy or "current_replay",
        "question": question or None,
    }
    try:
        resp = httpx.post(f"{API_BASE_URL}/analyze", json=payload, timeout=25.0)
        if resp.status_code == 200:
            data = resp.json()
            return f"*Source: `{data.get('source', 'api')}`*\n\n{data.get('markdown', '')}"
    except Exception:
        pass

    from app.evaluation.ai_analyst import analyze_experiment_with_gemini

    settings = load_app_settings()
    registry = ExperimentRegistry(settings.duckdb_path)
    exp_df = registry.get_kpi_dataframe(experiment_name=experiment_name)
    res = analyze_experiment_with_gemini(
        experiment_name=experiment_name,
        exp_df=exp_df,
        baseline_policy_id=baseline_policy or "current_replay",
        question=question or None,
    )
    return f"*Source: `{res.get('source', 'local')}`*\n\n{res.get('markdown', '')}"


def _run_interactive_backtest(
    exp_name: str,
    policies: list[str],
    scenarios: list[str],
    seeds_text: str,
    baseline_id: str,
) -> tuple[str, Any]:
    """Run multi-seed backtest via FastAPI when available, else lazy-load runner in-process."""
    if not policies or not scenarios:
        return "Select at least one policy and one scenario.", gr.update()
    seeds = [int(s.strip()) for s in seeds_text.split(",") if s.strip()] or [101, 202, 303, 404, 505]
    if baseline_id not in policies:
        baseline_id = policies[0]
    target_exp = exp_name.strip() or "gradio_backtest"

    # Prefer delegating heavy simulation batch to FastAPI server if up
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/backtest",
            json={
                "experiment_name": target_exp,
                "policy_ids": policies,
                "scenario_ids": scenarios,
                "seeds": seeds,
                "baseline_policy_id": baseline_id,
            },
            timeout=120.0,
        )
        if resp.status_code == 200:
            _VIEW_CACHE.clear()
            data = resp.json()
            return (
                f"Completed {data['run_count']} runs via FastAPI for `{target_exp}`.",
                gr.update(value=target_exp),
            )
    except Exception:
        pass

    # Lazy import heavy ML/Sim dependencies only when needed locally
    from app.data.synthetic import generate_synthetic_dataset
    from app.experiments.runner import run_backtest_suite

    settings = load_app_settings()
    scenarios_map = load_scenarios()
    ds = generate_synthetic_dataset(seed=42)
    out = run_backtest_suite(
        dataset=ds,
        scenarios=[scenarios_map[s] for s in scenarios],
        policy_ids=policies,
        seeds=seeds,
        settings=settings,
        experiment_name=target_exp,
        baseline_policy_id=baseline_id,
    )
    _VIEW_CACHE.clear()
    return (
        f"Completed {len(out['runs'])} runs for experiment `{out['experiment_name']}`.",
        gr.update(value=out["experiment_name"]),
    )


def build_demo() -> gr.Blocks:
    """Construct the fast, cached Gradio reverse-logistics backtester dashboard."""
    scenarios_map = load_scenarios()
    policy_choices = [p for p in available_policy_ids() if p != "do_nothing"]
    scenario_choices = list(scenarios_map.keys())

    with gr.Blocks(title="Logistics Backtester — Reverse Logistics Operations") as demo:
        gr.Markdown(
            "# Logistics Backtester — Reverse Logistics Operations\n"
            "Discrete-event SimPy simulation evaluating **Current Replay**, **Rule Heuristics**, "
            "**Google OR-Tools CP-SAT**, **ML Risk Scorer**, and **Stable-Baselines3 PPO**."
        )

        with gr.Row():
            exp_dd = gr.Dropdown(label="Experiment", choices=[], allow_custom_value=True)
            base_dd = gr.Dropdown(label="Paired Comparison Baseline", choices=[], allow_custom_value=True)
            refresh_btn = gr.Button("Refresh Registry", variant="secondary")

        banner_md = gr.Markdown("Loading experiment registry...")

        with gr.Tabs():
            with gr.Tab("Policy Comparison"):
                tradeoff_plot = gr.Plot(label="Cost vs. OTD Tradeoff")
                with gr.Accordion("Gemini AI Executive Analyst (On-Demand API)", open=False):
                    ai_q = gr.Textbox(
                        label="Question (optional)",
                        placeholder="e.g., Which policy gives the best cost/OTD tradeoff with zero constraint violations?",
                    )
                    ai_btn = gr.Button("Analyze Results with Gemini API", variant="primary")
                    ai_out = gr.Markdown("")

                gr.Markdown("### Mean Policy KPIs Across Scenarios & Seeds")
                agg_table = gr.Dataframe(interactive=False)
                gr.Markdown(
                    "### Paired Statistical Uplift vs. Baseline (t-test, Wilcoxon, Permutation, Holm, 95% CI, Cohen's d)"
                )
                comp_table = gr.Dataframe(interactive=False)

            with gr.Tab("Scenario Breakdown"):
                scen_plot = gr.Plot(label="Mean Cost by Scenario & Policy")
                gr.Markdown("### Per-Scenario Seed Distribution (Mean, Median, Std, P5, P95)")
                summary_table = gr.Dataframe(interactive=False)

            with gr.Tab("Run Drill-Down & Event Stream"):
                run_dd = gr.Dropdown(label="Select Run ID", choices=[], allow_custom_value=True)
                with gr.Row():
                    attr_json = gr.JSON(label="Cost & Bottleneck Attribution")
                    kpi_json = gr.JSON(label="Run KPI Metrics")
                ev_plot = gr.Plot(label="Immutable Event Distribution")
                ev_table = gr.Dataframe(label="Time-Ordered Event Log (top 500)", interactive=False)

            with gr.Tab("Run Backtest & Export Reports"):
                with gr.Row():
                    new_exp_in = gr.Textbox(label="New Experiment Name", value="gradio_reverse_run")
                    seeds_in = gr.Textbox(label="Seeds (comma-separated)", value="101,202,303,404,505")
                    new_base_dd = gr.Dropdown(
                        label="Baseline Policy",
                        choices=policy_choices,
                        value="current_replay",
                    )
                pol_chk = gr.CheckboxGroup(
                    label="Policies",
                    choices=policy_choices,
                    value=["current_replay", "fifo_edf", "greedy_cost", "ortools_optimizer"],
                )
                scen_chk = gr.CheckboxGroup(
                    label="Scenarios",
                    choices=scenario_choices,
                    value=scenario_choices[:3],
                )
                with gr.Row():
                    run_bt_btn = gr.Button("Run Multi-Seed Backtest", variant="primary")
                    gen_exp_btn = gr.Button("Generate / Refresh Export Bundle", variant="secondary")
                bt_status = gr.Markdown("")

                gr.Markdown("### Downloadable Reports & Result Sets")
                with gr.Row():
                    csv_file = gr.File(label="KPI Results (CSV)")
                    pq_file = gr.File(label="KPI Results (Parquet)")
                    html_file = gr.File(label="HTML Report")
                    pdf_file = gr.File(label="PDF Report")

        outputs_main = [
            banner_md,
            agg_table,
            tradeoff_plot,
            comp_table,
            scen_plot,
            summary_table,
            exp_dd,
            base_dd,
            run_dd,
            csv_file,
            pq_file,
            html_file,
            pdf_file,
        ]

        demo.load(fn=_load_experiment_view, inputs=[exp_dd, base_dd], outputs=outputs_main).then(
            fn=_inspect_run, inputs=[exp_dd, run_dd], outputs=[attr_json, kpi_json, ev_plot, ev_table]
        )
        refresh_btn.click(
            fn=lambda e, b: (_VIEW_CACHE.clear(), _load_experiment_view(e, b))[1],
            inputs=[exp_dd, base_dd],
            outputs=outputs_main,
        )
        exp_dd.change(fn=_load_experiment_view, inputs=[exp_dd, base_dd], outputs=outputs_main).then(
            fn=_inspect_run, inputs=[exp_dd, run_dd], outputs=[attr_json, kpi_json, ev_plot, ev_table]
        )
        base_dd.change(fn=_load_experiment_view, inputs=[exp_dd, base_dd], outputs=outputs_main)
        run_dd.change(fn=_inspect_run, inputs=[exp_dd, run_dd], outputs=[attr_json, kpi_json, ev_plot, ev_table])
        ai_btn.click(fn=_ask_ai_analyst, inputs=[exp_dd, base_dd, ai_q], outputs=[ai_out])
        gen_exp_btn.click(
            fn=_generate_exports_on_demand,
            inputs=[exp_dd, base_dd],
            outputs=[csv_file, pq_file, html_file, pdf_file, bt_status],
        )
        run_bt_btn.click(
            fn=_run_interactive_backtest,
            inputs=[new_exp_in, pol_chk, scen_chk, seeds_in, new_base_dd],
            outputs=[bt_status, exp_dd],
        ).then(fn=_load_experiment_view, inputs=[exp_dd, base_dd], outputs=outputs_main)

    return demo


def main() -> None:
    """Launch the Gradio dashboard server."""
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
