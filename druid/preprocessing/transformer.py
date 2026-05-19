"""
druid.preprocessing.transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature engineering transformations: datetime features, age binning,
outlier treatment, and custom transforms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer

from druid.core.dataset import DruidDataset


def featurize_datetime(
    dataset: DruidDataset,
    columns: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    drop_original: bool = True,
) -> DruidDataset:
    """
    Extract temporal features from datetime columns.

    Uses feature-engine's ``DatetimeFeatures`` transformer.

    Parameters
    ----------
    dataset : DruidDataset
    columns : list of str, optional
        Datetime columns to featurize.  Auto-detects if omitted.
    features : list of str, optional
        Which features to extract.  Defaults to a standard set.
    drop_original : bool
        Whether to drop the original datetime columns.

    Returns
    -------
    DruidDataset
    """
    df = dataset.df

    if columns is None:
        columns = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    if not columns:
        return dataset

    default_features = ["month", "day_of_week", "day_of_month", "hour"]
    features = features or default_features

    for col in columns:
        # Ensure datetime dtype
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    dtf = DatetimeFeatures(
        variables=columns,
        features_to_extract=features,
        drop_original=drop_original,
    )
    dataset.df = dtf.fit_transform(df)

    dataset.session.log(
        "featurize_datetime",
        params={"columns": columns, "features": features, "drop_original": drop_original},
        code=(
            f"from feature_engine.datetime import DatetimeFeatures\n"
            f"dtf = DatetimeFeatures(variables={columns}, "
            f"features_to_extract={features}, drop_original={drop_original})\n"
            f"df = dtf.fit_transform(df)"
        ),
    )

    return dataset


def bin_numeric(
    dataset: DruidDataset,
    column: str,
    bins: List[float],
    labels: List[str],
    output_column: Optional[str] = None,
    drop_original: bool = False,
) -> DruidDataset:
    """
    Bin a numeric column into categorical groups.

    Parameters
    ----------
    dataset : DruidDataset
    column : str
        Column to bin.
    bins : list of float
        Bin edges (e.g. ``[0, 18, 35, 65, 120]``).
    labels : list of str
        Labels for each bin (length must be ``len(bins) - 1``).
    output_column : str, optional
        Name for the new column.  Defaults to ``f"binned_{column}"``.
    drop_original : bool
        Drop the original column after binning.

    Returns
    -------
    DruidDataset
    """
    out_col = output_column or f"binned_{column}"
    dataset.df[out_col] = pd.cut(
        dataset.df[column], bins=bins, labels=labels, include_lowest=True,
    )

    if drop_original:
        dataset.df.drop(columns=[column], inplace=True)

    dataset.session.log(
        "bin_numeric",
        params={"column": column, "bins": bins, "labels": labels, "output": out_col},
    )

    return dataset


def treat_outliers(
    dataset: DruidDataset,
    columns: Optional[List[str]] = None,
    method: Optional[str] = None,
    fold: Optional[float] = None,
) -> DruidDataset:
    """
    Cap outliers using feature-engine's Winsorizer.

    Parameters
    ----------
    dataset : DruidDataset
    columns : list of str, optional
        Numeric columns to treat.  Auto-selects if omitted.
    method : str, optional
        ``"iqr"`` or ``"gaussian"``.  Defaults to config.
    fold : float, optional
        Multiplier for the fence.  Defaults to config.

    Returns
    -------
    DruidDataset
    """
    cfg = dataset.config.pipeline
    method = method or cfg.outlier_method
    fold = fold or cfg.outlier_fold
    target = dataset.target or ""

    if columns is None:
        columns = [
            c for c in dataset.df.select_dtypes(include=[np.number]).columns
            if c != target and dataset.df[c].nunique() > 10
        ]

    if not columns:
        return dataset

    winsorizer = Winsorizer(
        capping_method=method,
        tail="both",
        fold=fold,
        variables=columns,
    )
    dataset.df = winsorizer.fit_transform(dataset.df)

    dataset.session.log(
        "treat_outliers",
        params={"columns": columns, "method": method, "fold": fold},
        code=(
            f"from feature_engine.outliers import Winsorizer\n"
            f"w = Winsorizer(capping_method='{method}', fold={fold}, variables={columns})\n"
            f"df = w.fit_transform(df)"
        ),
    )

    return dataset
