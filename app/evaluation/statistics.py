"""Multi-seed aggregation, paired statistical significance, confidence intervals, and uplift."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

LOWER_IS_BETTER_METRICS = {
    "total_cost",
    "cost_per_order",
    "net_cost_after_recovery",
    "late_orders",
    "avg_cycle_time_minutes",
    "p95_cycle_time_minutes",
    "stockouts",
    "backorders",
    "empty_miles_km",
    "total_miles_km",
    "carbon_kg_co2",
    "constraint_violations",
    "exceptions_raised",
}

SUMMARY_METRICS = [
    "on_time_delivery_rate",
    "fill_rate",
    "total_cost",
    "cost_per_order",
    "net_cost_after_recovery",
    "resource_utilization",
    "avg_cycle_time_minutes",
    "empty_miles_km",
    "carbon_kg_co2",
    "constraint_violations",
]


def summarize_across_seeds(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median, P5, P95, and std across seeds per (scenario_id, policy_id)."""
    if results_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (scenario_id, policy_id), grp in results_df.groupby(["scenario_id", "policy_id"], sort=True):
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "policy_id": policy_id,
            "n_seeds": int(len(grp)),
        }
        for col in SUMMARY_METRICS:
            if col not in grp.columns:
                continue
            vals = grp[col].astype(float).to_numpy()
            row[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{col}_median"] = round(float(np.median(vals)), 4)
            row[f"{col}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            row[f"{col}_p5"] = round(float(np.percentile(vals, 5)), 4)
            row[f"{col}_p95"] = round(float(np.percentile(vals, 95)), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    bootstrap_seed: int = 2026,
) -> tuple[float, float]:
    """Compute reproducible paired bootstrap 95% CI."""
    if len(diffs) <= 1:
        val = float(diffs[0]) if len(diffs) == 1 else 0.0
        return val, val
    rng = np.random.default_rng(bootstrap_seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = np.mean(diffs[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _paired_permutation_p(
    diffs: np.ndarray,
    n_perm: int = 1000,
    seed: int = 2026,
) -> float:
    """Two-sided paired sign-flip permutation p-value."""
    if len(diffs) == 0 or np.allclose(diffs, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(diffs)))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = np.abs(np.mean(diffs * signs, axis=1))
    return float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction across multiple tests."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m
    prev = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        corr = min(1.0, max(prev, pval * (m - rank)))
        adjusted[orig_idx] = round(corr, 5)
        prev = corr
    return adjusted


def compare_policies_paired(
    results_df: pd.DataFrame,
    baseline_policy_id: str = "current_replay",
    metrics: tuple[str, ...] = ("total_cost", "on_time_delivery_rate", "resource_utilization", "empty_miles_km"),
    practical_threshold_pct: float = 3.0,
    bootstrap_seed: int = 2026,
) -> pd.DataFrame:
    """Perform paired across-seed comparisons (t-test, Wilcoxon, permutation, bootstrap CI, Holm correction, Cohen's d, uplift)."""
    if results_df.empty or baseline_policy_id not in set(results_df["policy_id"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    scenarios = sorted(results_df["scenario_id"].unique())
    policies = sorted(p for p in results_df["policy_id"].unique() if p != baseline_policy_id)

    for scenario_id in scenarios:
        base_df = results_df[
            (results_df["scenario_id"] == scenario_id)
            & (results_df["policy_id"] == baseline_policy_id)
        ].sort_values("seed")

        for policy_id in policies:
            cand_df = results_df[
                (results_df["scenario_id"] == scenario_id)
                & (results_df["policy_id"] == policy_id)
            ].sort_values("seed")

            merged = pd.merge(
                cand_df, base_df, on="seed", suffixes=("_cand", "_base"), how="inner"
            )
            if merged.empty:
                continue

            for metric in metrics:
                c_vals = merged[f"{metric}_cand"].astype(float).to_numpy()
                b_vals = merged[f"{metric}_base"].astype(float).to_numpy()
                # Define improvement > 0 when policy outperforms baseline
                diffs = (
                    b_vals - c_vals
                    if metric in LOWER_IS_BETTER_METRICS
                    else c_vals - b_vals
                )

                n = len(diffs)
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
                base_mean = float(np.mean(b_vals))
                cand_mean = float(np.mean(c_vals))
                uplift_pct = (mean_diff / abs(base_mean) * 100.0) if abs(base_mean) > 1e-9 else 0.0

                if n > 1 and std_diff > 1e-9:
                    se = std_diff / math.sqrt(n)
                    t_crit = float(stats.t.ppf(0.975, df=n - 1))
                    ci_low = mean_diff - t_crit * se
                    ci_high = mean_diff + t_crit * se
                    boot_low, boot_high = _paired_bootstrap_ci(diffs, bootstrap_seed=bootstrap_seed)
                    _, p_ttest = stats.ttest_rel(c_vals, b_vals)
                    try:
                        _, p_wilcox = stats.wilcoxon(diffs)
                    except ValueError:
                        p_wilcox = 1.0
                    p_perm = _paired_permutation_p(diffs, seed=bootstrap_seed)
                    cohens_d = mean_diff / std_diff
                else:
                    ci_low = mean_diff
                    ci_high = mean_diff
                    boot_low = mean_diff
                    boot_high = mean_diff
                    p_ttest = 1.0 if abs(mean_diff) < 1e-9 else 0.0
                    p_wilcox = p_ttest
                    p_perm = p_ttest
                    cohens_d = 0.0

                stat_sig = bool(float(p_ttest) < 0.05)
                pract_sig = bool(abs(uplift_pct) >= practical_threshold_pct)
                if mean_diff > 0 and stat_sig and pract_sig:
                    verdict = "IMPROVED_SIGNIFICANT"
                elif mean_diff > 0 and pract_sig:
                    verdict = "IMPROVED_PRACTICAL"
                elif mean_diff < 0 and stat_sig and pract_sig:
                    verdict = "REGRESSED_SIGNIFICANT"
                else:
                    verdict = "EQUIVALENT"

                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "policy_id": policy_id,
                        "baseline_policy_id": baseline_policy_id,
                        "metric": metric,
                        "n_pairs": n,
                        "policy_mean": round(cand_mean, 4),
                        "baseline_mean": round(base_mean, 4),
                        "uplift_abs": round(mean_diff, 4),
                        "uplift_pct": round(uplift_pct, 2),
                        "ci_95_low": round(ci_low, 4),
                        "ci_95_high": round(ci_high, 4),
                        "boot_ci_95_low": round(boot_low, 4),
                        "boot_ci_95_high": round(boot_high, 4),
                        "p_value_ttest": round(float(p_ttest), 5),
                        "p_value_wilcoxon": round(float(p_wilcox), 5),
                        "p_value_permutation": round(float(p_perm), 5),
                        "cohens_d": round(float(cohens_d), 3),
                        "statistically_significant": stat_sig,
                        "practically_significant": pract_sig,
                        "verdict": verdict,
                    }
                )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df["p_value_holm"] = _holm_bonferroni(out_df["p_value_ttest"].tolist())
    return out_df
