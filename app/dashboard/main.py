"""Streamlit MVP dashboard for policy comparison, scenario comparison, and run drill-down."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from app.config import load_app_settings
from app.data.synthetic import generate_synthetic_dataset
from app.domain.entities import KPIResult
from app.evaluation.attribution import compute_cost_and_bottleneck_attribution
from app.evaluation.reporting import (
    export_results_csv,
    export_results_parquet,
    generate_html_report,
    generate_pdf_report,
)
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds
from app.experiments.registry import ExperimentRegistry
from app.experiments.runner import run_backtest_suite
from app.experiments.scenarios import load_scenarios
from app.policies.registry import available_policy_ids

st.set_page_config(
    page_title="Logistics Backtester — Reverse Logistics MVP",
    layout="wide",
)

st.title("Logistics Backtester — Reverse Logistics Operations")
st.caption("Evaluate AI, OR-Tools optimization, and rule policies on SimPy discrete-event simulations.")

settings = load_app_settings()
registry = ExperimentRegistry(settings.duckdb_path)

with st.sidebar:
    st.header("Run Quick Backtest")
    exp_name = st.text_input("Experiment Name", value="reverse_mvp_backtest")
    selected_policies = st.multiselect(
        "Policies",
        options=[p for p in available_policy_ids() if p != "do_nothing"],
        default=["current_replay", "fifo_edf", "greedy_cost", "ortools_optimizer"],
    )
    scenarios_map = load_scenarios()
    selected_scenarios = st.multiselect(
        "Scenarios",
        options=list(scenarios_map.keys()),
        default=list(scenarios_map.keys())[:3],
    )
    seeds_text = st.text_input("Seeds (comma-separated)", value="101,202,303,404,505")
    baseline_id = st.selectbox("Baseline Policy", options=selected_policies or ["current_replay"])

    if st.button("Run Backtest Suite", type="primary"):
        seeds = [int(s.strip()) for s in seeds_text.split(",") if s.strip()]
        ds = generate_synthetic_dataset(seed=42)
        with st.spinner("Running discrete-event simulations across seeds & scenarios..."):
            run_backtest_suite(
                dataset=ds,
                scenarios=[scenarios_map[s] for s in selected_scenarios],
                policy_ids=selected_policies,
                seeds=seeds,
                settings=settings,
                experiment_name=exp_name,
                baseline_policy_id=baseline_id,
            )
        st.success(f"Completed backtest '{exp_name}'.")

df = registry.get_kpi_dataframe()
if df.empty:
    st.info("No backtest runs recorded yet. Use the sidebar or `logisticsbt backtest` to populate results.")
    st.stop()

exp_options = sorted(df["experiment_name"].unique().tolist())
chosen_exp = st.selectbox("Select Experiment", options=exp_options, index=len(exp_options) - 1)
exp_df = df[df["experiment_name"] == chosen_exp].copy()
base_choice = st.selectbox(
    "Paired Comparison Baseline",
    options=sorted(exp_df["policy_id"].unique().tolist()),
    index=0,
)

summary_df = summarize_across_seeds(exp_df)
comp_df = compare_policies_paired(exp_df, baseline_policy_id=base_choice)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Policy Comparison", "Scenario Breakdown", "Run Drill-Down & Events", "Exports & Reports"]
)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded Runs", len(exp_df))
    c2.metric("Policies", exp_df["policy_id"].nunique())
    c3.metric("Scenarios", exp_df["scenario_id"].nunique())
    c4.metric("Best Mean OTD", f"{exp_df.groupby('policy_id')['on_time_delivery_rate'].mean().max():.1%}")

    fig_tradeoff = px.scatter(
        exp_df.groupby("policy_id", as_index=False)[
            ["total_cost", "on_time_delivery_rate", "resource_utilization", "constraint_violations"]
        ].mean(),
        x="total_cost",
        y="on_time_delivery_rate",
        size="resource_utilization",
        color="policy_id",
        title="Cost vs. On-Time Delivery Rate (Mean Across Scenarios & Seeds)",
    )
    st.plotly_chart(fig_tradeoff, width="stretch")

    st.subheader("Paired Statistical Uplift vs Baseline")
    st.dataframe(comp_df, width="stretch")

with tab2:
    fig_scen = px.bar(
        exp_df.groupby(["scenario_id", "policy_id"], as_index=False)["total_cost"].mean(),
        x="scenario_id",
        y="total_cost",
        color="policy_id",
        barmode="group",
        title="Mean Total Cost by Scenario & Policy",
    )
    st.plotly_chart(fig_scen, width="stretch")
    st.subheader("Per-Scenario Distribution (Mean, Median, Std, P5, P95)")
    st.dataframe(summary_df, width="stretch")

with tab3:
    run_choice = st.selectbox("Inspect Run ID", options=exp_df["run_id"].tolist())
    run_row = exp_df[exp_df["run_id"] == run_choice].iloc[0]
    runs_meta = registry.list_runs(experiment_name=chosen_exp, limit=500)
    matched = [r for r in runs_meta if r.get("run_id") == run_choice]
    if matched and "metrics_json" in matched[0]:
        kpi_obj = KPIResult.model_validate_json(str(matched[0]["metrics_json"]))
        st.subheader("Cost & Bottleneck Attribution")
        st.json(compute_cost_and_bottleneck_attribution(kpi_obj))
    st.subheader("Run KPI Metrics")
    st.json(run_row.to_dict())
    events_df = registry.get_run_events(run_choice, limit=500)
    if not events_df.empty:
        ev_counts = events_df["event_type"].value_counts().reset_index()
        ev_counts.columns = ["event_type", "count"]
        st.plotly_chart(
            px.bar(ev_counts, x="event_type", y="count", title="Immutable Event Stream Distribution"),
            width="stretch",
        )
        st.dataframe(events_df, width="stretch")
    else:
        st.info("Event rows for non-primary seeds are archived to Parquet on disk to keep DuckDB compact.")

with tab4:
    rep_dir = settings.artifacts_dir / "reports" / chosen_exp
    csv_p = export_results_csv(exp_df, rep_dir / "kpi_results.csv")
    pq_p = export_results_parquet(exp_df, rep_dir / "kpi_results.parquet")
    html_p = generate_html_report(exp_df, summary_df, comp_df, rep_dir / "report.html", base_choice)
    pdf_p = generate_pdf_report(exp_df, summary_df, comp_df, rep_dir / "report.pdf", base_choice)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.download_button("Download CSV", csv_p.read_bytes(), file_name=f"{chosen_exp}.csv")
    col_b.download_button("Download Parquet", pq_p.read_bytes(), file_name=f"{chosen_exp}.parquet")
    col_c.download_button("Download HTML Report", html_p.read_bytes(), file_name=f"{chosen_exp}.html")
    col_d.download_button("Download PDF Report", pdf_p.read_bytes(), file_name=f"{chosen_exp}.pdf")
