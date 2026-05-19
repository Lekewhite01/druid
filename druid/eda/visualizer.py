"""
druid.eda.visualizer
~~~~~~~~~~~~~~~~~~~~

Automated visualisation engine.  Generates appropriate plots
based on column types, distributions, and the target variable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from druid.core.dataset import DruidDataset


def plot_missing_values(dataset: DruidDataset, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Bar chart of missing value percentages per column.

    Parameters
    ----------
    dataset : DruidDataset
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    missing = dataset.df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]

    if len(missing) == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, len(missing) * 0.3)))
    else:
        fig = ax.figure

    colors = ["#e74c3c" if v > 0.5 else "#f39c12" if v > 0.2 else "#27ae60" for v in missing]
    ax.barh(missing.index, missing.values * 100, color=colors)
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column")
    ax.invert_yaxis()

    # Add percentage labels
    for i, (col, val) in enumerate(missing.items()):
        ax.text(val * 100 + 0.5, i, f"{val*100:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def plot_distributions(
    dataset: DruidDataset,
    columns: Optional[List[str]] = None,
    max_cols: int = 12,
) -> plt.Figure:
    """
    Histogram / bar plots for numeric and categorical columns.

    Parameters
    ----------
    dataset : DruidDataset
    columns : list of str, optional
        Specific columns to plot.  If omitted, selects up to
        *max_cols* most interesting columns.
    max_cols : int
        Maximum number of subplots.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = dataset.df

    if columns is None:
        # Pick a mix of numeric and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Prioritise columns with variance
        columns = num_cols[:max_cols // 2] + cat_cols[:max_cols // 2]
        columns = columns[:max_cols]

    if not columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No plottable columns", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if len(columns) > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        series = df[col].dropna()

        if pd.api.types.is_numeric_dtype(series):
            ax.hist(series, bins=30, edgecolor="white", alpha=0.8, color="#3498db")
            ax.axvline(series.mean(), color="#e74c3c", linestyle="--", label=f"mean={series.mean():.2f}")
            ax.legend(fontsize=8)
        else:
            vc = series.value_counts().head(10)
            ax.barh(vc.index.astype(str), vc.values, color="#2ecc71")
            ax.invert_yaxis()

        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distributions — {dataset.name}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_target_analysis(
    dataset: DruidDataset,
    top_n_features: int = 6,
) -> Optional[plt.Figure]:
    """
    Visualise the relationship between the target and top features.

    For classification: grouped box plots / count plots.
    For regression: scatter plots with trend lines.

    Parameters
    ----------
    dataset : DruidDataset
        Must have ``target`` set.
    top_n_features : int
        Number of features to show.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if dataset.target is None or dataset.target not in dataset.df.columns:
        return None

    df = dataset.df
    target = dataset.target
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target
    ]

    if not num_cols:
        return None

    # Select top features by correlation with target
    if pd.api.types.is_numeric_dtype(df[target]):
        corrs = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        top_feats = corrs.head(top_n_features).index.tolist()
    else:
        top_feats = num_cols[:top_n_features]

    n_cols_plot = min(3, len(top_feats))
    n_rows = (len(top_feats) + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(5 * n_cols_plot, 4 * n_rows))
    axes = np.array(axes).flatten() if len(top_feats) > 1 else [axes]

    is_classification = df[target].nunique() <= 20

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        if is_classification:
            # Box plot grouped by target
            for label in sorted(df[target].dropna().unique()):
                subset = df[df[target] == label][feat].dropna()
                ax.hist(subset, bins=25, alpha=0.5, label=str(label))
            ax.legend(title=target, fontsize=7)
        else:
            ax.scatter(df[feat], df[target], alpha=0.3, s=5, color="#3498db")
            ax.set_ylabel(target)

        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Target Analysis — {target}", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    dataset: DruidDataset,
    max_cols: int = 20,
) -> plt.Figure:
    """
    Heatmap of pairwise correlations for numeric columns.

    Parameters
    ----------
    dataset : DruidDataset
    max_cols : int
        Maximum columns to include (selects most variable).

    Returns
    -------
    matplotlib.figure.Figure
    """
    num_df = dataset.df.select_dtypes(include=[np.number])

    if num_df.shape[1] > max_cols:
        # Keep the most variable columns
        variances = num_df.var().sort_values(ascending=False)
        num_df = num_df[variances.head(max_cols).index]

    if num_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough numeric columns for correlation", ha="center", va="center")
        ax.set_axis_off()
        return fig

    corr = num_df.corr()
    size = max(8, num_df.shape[1] * 0.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.8))

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=num_df.shape[1] <= 15,
        fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
        square=True, linewidths=0.5,
        annot_kws={"size": 7},
    )
    ax.set_title(f"Correlation Heatmap — {dataset.name}", fontsize=12)
    plt.tight_layout()
    return fig


def auto_eda_plots(
    dataset: DruidDataset,
    save_dir: Optional[str] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate a full suite of EDA visualisations.

    Parameters
    ----------
    dataset : DruidDataset
    save_dir : str, optional
        If provided, save all plots as PNG files to this directory.

    Returns
    -------
    dict
        Plot name → Figure mapping.
    """
    import os

    plots: Dict[str, plt.Figure] = {}

    plots["missing_values"] = plot_missing_values(dataset)
    plots["distributions"] = plot_distributions(dataset)
    plots["correlation_heatmap"] = plot_correlation_heatmap(dataset)

    target_fig = plot_target_analysis(dataset)
    if target_fig is not None:
        plots["target_analysis"] = target_fig

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in plots.items():
            fig.savefig(
                os.path.join(save_dir, f"{name}.png"),
                dpi=150, bbox_inches="tight",
            )

    # Log
    dataset.session.log(
        "auto_eda_plots",
        params={"plots": list(plots.keys()), "save_dir": save_dir},
    )

    return plots
