"""On-demand Gemini API executive summary & policy uplift analyst (uses google-genai)."""

from __future__ import annotations

import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from app.evaluation.statistics import compare_policies_paired

GEMINI_TIMEOUT_SEC = 8.0


def _load_api_key() -> str | None:
    """Load GEMINI_API_KEY from environment or gitignored .env without logging secrets."""
    for candidate in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        if candidate.exists():
            load_dotenv(dotenv_path=candidate, override=False)
            break
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    return key or None


def _deterministic_fallback_summary(
    experiment_name: str,
    baseline_policy_id: str,
    agg_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    note: str | None = None,
) -> str:
    """Produce a transparent, non-LLM summary strictly from observed KPI tables."""
    if agg_df.empty:
        return "No experiment runs available to analyze."
    best_cost = agg_df.sort_values("total_cost").iloc[0]
    best_otd = agg_df.sort_values("on_time_delivery_rate", ascending=False).iloc[0]
    zero_viol = agg_df[agg_df["constraint_violations"] == 0]
    best_safe = (
        zero_viol.sort_values("net_cost_after_recovery").iloc[0]
        if not zero_viol.empty
        else best_cost
    )
    sig_count = (
        int((comp_df["verdict"] == "IMPROVED_SIGNIFICANT").sum())
        if not comp_df.empty and "verdict" in comp_df.columns
        else 0
    )
    lines = [
        f"### Quantitative Summary for `{experiment_name}` (Baseline: `{baseline_policy_id}`)",
        f"- **Lowest Gross Cost:** `{best_cost['policy_id']}` at **${best_cost['total_cost']:,.2f}** "
        f"(OTD `{best_cost['on_time_delivery_rate']:.1%}`, violations `{best_cost['constraint_violations']:.2f}`).",
        f"- **Best On-Time Delivery:** `{best_otd['policy_id']}` at **{best_otd['on_time_delivery_rate']:.1%}** "
        f"(gross cost `${best_otd['total_cost']:,.2f}`, violations `{best_otd['constraint_violations']:.2f}`).",
        f"- **Best Zero-Violation Net Cost:** `{best_safe['policy_id']}` at **${best_safe['net_cost_after_recovery']:,.2f}** net after recovery.",
        f"- **Statistically & Practically Significant Improvements vs. `{baseline_policy_id}`:** `{sig_count}` metric-scenario pairs.",
    ]
    if note:
        lines.append(f"\n> *Gemini API unavailable ({note}); displaying deterministic quantitative fallback.*")
    return "\n".join(lines)


def analyze_experiment_with_gemini(
    experiment_name: str,
    exp_df: pd.DataFrame,
    baseline_policy_id: str = "current_replay",
    question: str | None = None,
    model: str = "gemini-4-hydrogen",
    timeout_sec: float = GEMINI_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Generate an executive comparison narrative via Gemini API (`google-genai`) with strict timeout."""
    if exp_df.empty:
        return {"source": "none", "markdown": "No runs found for this experiment."}

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
    agg_df = exp_df.groupby("policy_id", as_index=False)[mean_cols].mean().round(3)
    comp_df = compare_policies_paired(exp_df, baseline_policy_id=baseline_policy_id)

    api_key = _load_api_key()
    if not api_key:
        return {
            "source": "deterministic_fallback",
            "markdown": _deterministic_fallback_summary(
                experiment_name,
                baseline_policy_id,
                agg_df,
                comp_df,
                note="GEMINI_API_KEY not set in .env",
            ),
        }

    prompt_table = agg_df.to_csv(index=False)
    comp_subset = (
        comp_df[
            [
                "scenario_id",
                "policy_id",
                "metric",
                "uplift_pct",
                "p_value_ttest",
                "p_value_holm",
                "cohens_d",
                "verdict",
            ]
        ].to_csv(index=False)
        if not comp_df.empty
        else "No paired comparisons."
    )
    user_q = (question or "").strip() or (
        "Answer the core evaluation question: Would the AI/RL or optimization policy improve cost, "
        "service level, utilization, and resilience compared with current rules? Recommend the safest "
        "production candidate and highlight any constraint-violation or empty-mile tradeoffs."
    )

    def _call_gemini() -> str:
        from google import genai

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        try:
            sig = inspect.signature(genai.Client)
            if "http_options" in sig.parameters:
                client_kwargs["http_options"] = {"timeout": int(timeout_sec * 1000)}
        except (TypeError, ValueError):
            pass

        client = genai.Client(**client_kwargs)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "system_instruction": (
                "You are a senior logistics architect reviewing empirical discrete-event backtest results. "
                "Cite ONLY numbers directly present in the provided CSV tables. Never invent metrics, "
                "and explicitly call out constraint violations and statistical significance."
            ),
            "input": (
                f"Experiment: {experiment_name}\nBaseline: {baseline_policy_id}\n\n"
                f"Mean Policy KPIs:\n{prompt_table}\n\n"
                f"Paired Uplift & Significance:\n{comp_subset}\n\n"
                f"Question: {user_q}"
            ),
            "store": False,
        }
        try:
            create_sig = inspect.signature(client.interactions.create)
            if "timeout" in create_sig.parameters:
                create_kwargs["timeout"] = timeout_sec
        except (TypeError, ValueError):
            pass

        interaction = client.interactions.create(**create_kwargs)
        return str(getattr(interaction, "output_text", "") or "").strip()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call_gemini)
    try:
        text = future.result(timeout=timeout_sec)
        executor.shutdown(wait=False)
        if text:
            return {"source": f"gemini:{model}", "markdown": text}
    except FuturesTimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "source": "deterministic_fallback",
            "markdown": _deterministic_fallback_summary(
                experiment_name,
                baseline_policy_id,
                agg_df,
                comp_df,
                note=f"TimeoutError ({timeout_sec:.0f}s limit)",
            ),
        }
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        return {
            "source": "deterministic_fallback",
            "markdown": _deterministic_fallback_summary(
                experiment_name,
                baseline_policy_id,
                agg_df,
                comp_df,
                note=type(exc).__name__,
            ),
        }

    return {
        "source": "deterministic_fallback",
        "markdown": _deterministic_fallback_summary(
            experiment_name, baseline_policy_id, agg_df, comp_df, note="empty response"
        ),
    }
