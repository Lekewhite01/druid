"""
druid.eda
~~~~~~~~~

Exploratory data analysis: profiling, visualisation, and AI insights.
"""

from druid.eda.profiler import (
    compute_correlations,
    compute_distributions,
    compute_target_stats,
    full_profile,
)
from druid.eda.visualizer import (
    auto_eda_plots,
    plot_correlation_heatmap,
    plot_distributions,
    plot_missing_values,
    plot_target_analysis,
)
from druid.eda.insights import generate_eda_guidance, generate_insights

__all__ = [
    "compute_distributions",
    "compute_correlations",
    "compute_target_stats",
    "full_profile",
    "auto_eda_plots",
    "plot_correlation_heatmap",
    "plot_distributions",
    "plot_missing_values",
    "plot_target_analysis",
    "generate_eda_guidance",
    "generate_insights",
]
