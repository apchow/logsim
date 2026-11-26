"""Downloadable CSV, Parquet, HTML, and PDF comparison report generators."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from pathlib import Path

import pandas as pd
import polars as pl
from fpdf import FPDF
from fpdf.enums import XPos, YPos


def export_results_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export tabular backtest results to CSV."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def export_results_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Export tabular backtest results to Parquet via Polars."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(df).write_parquet(target)
    return target


def generate_html_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    baseline_policy_id: str = "current_replay",
) -> Path:
    """Generate a self-contained HTML comparison report with per-scenario & statistical tables."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    overall_cols = [
        "policy_id",
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
    available_cols = [c for c in overall_cols if c in results_df.columns]
    overall_agg = (
        results_df.groupby("policy_id", as_index=False)[available_cols[1:]].mean().round(3)
        if not results_df.empty
        else pd.DataFrame()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Logistics Backtester — Reverse Logistics Evaluation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 28px; color: #111827; background: #f8fafc; }}
    h1, h2, h3 {{ color: #0f172a; }}
    .meta {{ color: #475569; font-size: 0.92rem; margin-bottom: 20px; }}
    .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 22px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-top: 8px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 7px 9px; text-align: right; }}
    th {{ background: #f1f5f9; font-weight: 600; text-align: center; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr:nth-child(even) {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Logistics Backtester — Reverse Logistics Comparison Report</h1>
  <div class="meta">
    Generated: {escape(generated)} | Baseline Policy: <strong>{escape(baseline_policy_id)}</strong> |
    Total Runs: {len(results_df)} | Practical Significance Threshold: >= 3.0% uplift
  </div>

  <div class="card">
    <h2>1. Overall Policy Comparison (Mean Across All Scenarios &amp; Seeds)</h2>
    {overall_agg.to_html(index=False, border=0) if not overall_agg.empty else "<p>No runs found.</p>"}
  </div>

  <div class="card">
    <h2>2. Policy vs. Baseline Paired Statistical Uplift</h2>
    {comparison_df.to_html(index=False, border=0) if not comparison_df.empty else "<p>Baseline not present for paired comparison.</p>"}
  </div>

  <div class="card">
    <h2>3. Per-Scenario Distribution Summary (Mean, Median, Std, P5, P95)</h2>
    {summary_df.to_html(index=False, border=0) if not summary_df.empty else "<p>No summary data.</p>"}
  </div>
</body>
</html>
"""
    target.write_text(html, encoding="utf-8")
    return target


def generate_pdf_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    output_path: str | Path,
    baseline_policy_id: str = "current_replay",
) -> Path:
    """Generate a clean downloadable PDF comparison report using fpdf2."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(
        0,
        10,
        "Logistics Backtester - Reverse Logistics Evaluation Report",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        7,
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} | Baseline: {baseline_policy_id} | Runs: {len(results_df)}",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(3)

    def _render_table(title: str, df: pd.DataFrame, cols: list[str], col_w: int = 34) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if df.empty:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 6, "No rows available.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(3)
            return
        use_cols = [c for c in cols if c in df.columns]
        pdf.set_font("Helvetica", "B", 8)
        for c in use_cols:
            pdf.cell(col_w, 6, c[:18], border=1, align="C")
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for _, row in df.head(40).iterrows():
            for c in use_cols:
                val = row[c]
                txt = f"{val:.3f}" if isinstance(val, float) else str(val)
                pdf.cell(col_w, 6, txt[:18], border=1, align="L" if isinstance(val, str) else "R")
            pdf.ln()
        pdf.ln(4)

    if not results_df.empty:
        agg = (
            results_df.groupby("policy_id", as_index=False)[
                [
                    "on_time_delivery_rate",
                    "fill_rate",
                    "total_cost",
                    "cost_per_order",
                    "resource_utilization",
                    "empty_miles_km",
                    "constraint_violations",
                ]
            ]
            .mean()
            .round(3)
        )
        _render_table(
            "1. Overall Policy Comparison (Mean Across Scenarios & Seeds)",
            agg,
            list(agg.columns),
            col_w=34,
        )

    _render_table(
        "2. Policy vs. Baseline Paired Uplift & Significance",
        comparison_df,
        [
            "scenario_id",
            "policy_id",
            "metric",
            "policy_mean",
            "baseline_mean",
            "uplift_pct",
            "p_value_ttest",
            "cohens_d",
            "verdict",
        ],
        col_w=30,
    )

    _render_table(
        "3. Per-Scenario Seed Distribution (Mean / Median / Std / P5 / P95)",
        summary_df,
        [
            "scenario_id",
            "policy_id",
            "n_seeds",
            "total_cost_mean",
            "total_cost_p5",
            "total_cost_p95",
            "on_time_delivery_rate_mean",
            "constraint_violations_mean",
        ],
        col_w=33,
    )

    pdf.output(str(target))
    return target
