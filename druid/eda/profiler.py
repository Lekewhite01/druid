"""
druid.eda.profiler
~~~~~~~~~~~~~~~~~~

Statistical profiling — compute distributions, correlations,
and quality metrics for every column in the dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from druid.core.dataset import DruidDataset


def compute_distributions(dataset: DruidDataset) -> Dict[str, Dict[str, Any]]:
    """
    Compute distribution statistics for all columns.

    Parameters
    ----------
    dataset : DruidDataset

    Returns
    -------
    dict
        Column name → distribution summary.
    """
    df = dataset.df
    result: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "nunique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                info.update({
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                    "min": float(clean.min()),
                    "q25": float(clean.quantile(0.25)),
                    "median": float(clean.median()),
                    "q75": float(clean.quantile(0.75)),
                    "max": float(clean.max()),
                    "skew": float(clean.skew()),
                    "kurtosis": float(clean.kurtosis()),
                    "zeros_pct": round((clean == 0).mean() * 100, 2),
                })

        elif pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                info.update({
                    "min": str(clean.min()),
                    "max": str(clean.max()),
                    "range_days": (clean.max() - clean.min()).days,
                })

        else:
            # Categorical / object
            vc = series.value_counts()
            info.update({
                "top_5": vc.head(5).to_dict(),
                "bottom_5": vc.tail(5).to_dict() if len(vc) > 5 else {},
                "mode": str(vc.index[0]) if len(vc) > 0 else None,
                "mode_pct": round(vc.iloc[0] / len(df) * 100, 2) if len(vc) > 0 else 0,
            })

        result[col] = info

    return result


def compute_correlations(
    dataset: DruidDataset,
    method: str = "pearson",
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compute a correlation matrix for numeric columns and flag
    highly correlated pairs.

    Parameters
    ----------
    dataset : DruidDataset
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    threshold : float
        Absolute correlation above which a pair is flagged.

    Returns
    -------
    dict
        Keys: ``matrix`` (dict of dicts), ``high_pairs`` (list of tuples).
    """
    df = dataset.df.select_dtypes(include=[np.number])
    if df.shape[1] < 2:
        return {"matrix": {}, "high_pairs": []}

    corr = df.corr(method=method)

    # Find highly correlated pairs (upper triangle only)
    high_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = abs(corr.iloc[i, j])
            if val >= threshold:
                high_pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(corr.iloc[i, j]), 4),
                })

    # Sort by absolute correlation descending
    high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "matrix": corr.round(4).to_dict(),
        "high_pairs": high_pairs,
    }


def compute_target_stats(
    dataset: DruidDataset,
) -> Optional[Dict[str, Any]]:
    """
    Compute target-variable statistics.

    Parameters
    ----------
    dataset : DruidDataset
        Must have ``target`` set.

    Returns
    -------
    dict or None
        Target statistics, or None if no target is set.
    """
    if dataset.target is None or dataset.target not in dataset.df.columns:
        return None

    target = dataset.df[dataset.target]
    info: Dict[str, Any] = {
        "name": dataset.target,
        "dtype": str(target.dtype),
        "null_count": int(target.isna().sum()),
    }

    if pd.api.types.is_numeric_dtype(target):
        nunique = target.nunique()
        if nunique <= 20:
            # Classification
            info["task_type"] = "classification"
            vc = target.value_counts()
            info["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
            info["n_classes"] = nunique
            info["is_imbalanced"] = (vc.min() / vc.max()) < 0.1 if len(vc) > 1 else False
            if info["is_imbalanced"]:
                info["imbalance_ratio"] = round(vc.max() / max(vc.min(), 1), 1)
        else:
            # Regression
            info["task_type"] = "regression"
            info["mean"] = float(target.mean())
            info["std"] = float(target.std())
            info["skew"] = float(target.skew())
    else:
        # Categorical target
        info["task_type"] = "classification"
        vc = target.value_counts()
        info["class_distribution"] = {str(k): int(v) for k, v in vc.items()}
        info["n_classes"] = target.nunique()

    return info


def full_profile(dataset: DruidDataset) -> Dict[str, Any]:
    """
    Run the complete profiling suite.

    Parameters
    ----------
    dataset : DruidDataset

    Returns
    -------
    dict
        Combined profile with distributions, correlations,
        and target statistics.
    """
    profile = dataset.profile(force=True)
    profile["distributions"] = compute_distributions(dataset)
    profile["correlations"] = compute_correlations(dataset)
    profile["target_stats"] = compute_target_stats(dataset)

    # Log
    dataset.session.log(
        "full_profile",
        params={"n_cols": len(profile["distributions"])},
    )

    return profile
