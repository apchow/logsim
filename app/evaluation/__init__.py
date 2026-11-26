"""KPI computation, statistical analysis, attribution, and reporting."""

from app.evaluation.ai_analyst import analyze_experiment_with_gemini
from app.evaluation.attribution import compute_cost_and_bottleneck_attribution
from app.evaluation.kpi import compute_kpis
from app.evaluation.reporting import (
    export_results_csv,
    export_results_parquet,
    generate_html_report,
    generate_pdf_report,
)
from app.evaluation.statistics import compare_policies_paired, summarize_across_seeds

__all__ = [
    "analyze_experiment_with_gemini",
    "compare_policies_paired",
    "compute_cost_and_bottleneck_attribution",
    "compute_kpis",
    "export_results_csv",
    "export_results_parquet",
    "generate_html_report",
    "generate_pdf_report",
    "summarize_across_seeds",
]
