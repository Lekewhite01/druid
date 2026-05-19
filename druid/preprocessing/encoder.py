"""
druid.preprocessing.encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Categorical encoding: one-hot, frequency, and target encoding
with automatic cardinality-based strategy selection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from druid.core.dataset import DruidDataset


def encode_categoricals(
    dataset: DruidDataset,
    high_card_threshold: Optional[int] = None,
    drop_first: bool = True,
) -> DruidDataset:
    """
    Encode categorical columns automatically.

    - Low-cardinality columns (≤ threshold) → one-hot encoding.
    - High-cardinality columns (> threshold) → frequency encoding.

    Parameters
    ----------
    dataset : DruidDataset
    high_card_threshold : int, optional
        Cardinality threshold.  Defaults to config.
    drop_first : bool
        Drop the first dummy column to avoid multicollinearity.

    Returns
    -------
    DruidDataset
    """
    thresh = high_card_threshold or dataset.config.pipeline.high_cardinality_threshold
    cat_cols = dataset.df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        return dataset

    low_card = [c for c in cat_cols if dataset.df[c].nunique() <= thresh]
    high_card = [c for c in cat_cols if dataset.df[c].nunique() > thresh]

    # Frequency encode high-cardinality columns
    for col in high_card:
        freq = dataset.df[col].value_counts(normalize=True)
        dataset.df[f"{col}_freq"] = dataset.df[col].map(freq).astype(float)
    if high_card:
        dataset.df.drop(columns=high_card, inplace=True)

    # One-hot encode low-cardinality columns
    if low_card:
        dataset.df = pd.get_dummies(
            dataset.df, columns=low_card, drop_first=drop_first, dtype=int,
        )

    dataset.session.log(
        "encode_categoricals",
        params={
            "low_cardinality_ohe": low_card,
            "high_cardinality_freq": high_card,
            "threshold": thresh,
        },
    )

    return dataset


def target_encode(
    dataset: DruidDataset,
    columns: List[str],
    smoothing: float = 1.0,
) -> DruidDataset:
    """
    Target-encode specified columns (mean of target per category).

    Uses additive smoothing to avoid overfitting on rare categories.

    Parameters
    ----------
    dataset : DruidDataset
        Must have ``target`` set.
    columns : list of str
        Columns to encode.
    smoothing : float
        Smoothing factor (higher = more regularisation).

    Returns
    -------
    DruidDataset
    """
    if dataset.target is None:
        raise ValueError("Target must be set for target encoding")

    global_mean = dataset.df[dataset.target].mean()

    for col in columns:
        stats = dataset.df.groupby(col)[dataset.target].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        dataset.df[f"{col}_target_enc"] = dataset.df[col].map(smooth).astype(float)
        dataset.df.drop(columns=[col], inplace=True)

    dataset.session.log(
        "target_encode",
        params={"columns": columns, "smoothing": smoothing},
    )

    return dataset
