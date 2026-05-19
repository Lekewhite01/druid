"""
druid.preprocessing.pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

End-to-end preprocessing pipeline builder.  Chains cleaning,
feature engineering, encoding, and scaling into a single
reproducible workflow.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from druid.core.dataset import DruidDataset
from druid.preprocessing.cleaner import auto_clean
from druid.preprocessing.encoder import encode_categoricals
from druid.preprocessing.transformer import featurize_datetime, treat_outliers


def prepare(
    dataset: DruidDataset,
    auto: bool = True,
) -> DruidDataset:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    dataset : DruidDataset
    auto : bool
        If True, runs cleaning, datetime featurization,
        outlier treatment, and encoding automatically.

    Returns
    -------
    DruidDataset
        Preprocessed dataset ready for modelling.
    """
    if auto:
        dataset = auto_clean(dataset)
        dataset = featurize_datetime(dataset)
        dataset = treat_outliers(dataset)
        dataset = encode_categoricals(dataset)

    # Final dtype cleanup — ensure all numeric
    remaining = dataset.df.select_dtypes(
        include=["object", "category", "datetime64"]
    ).columns.tolist()
    if remaining:
        dataset.df.drop(columns=remaining, inplace=True)

    # Convert nullable ints to float
    for col in dataset.df.columns:
        if dataset.df[col].dtype.name.startswith("Int"):
            dataset.df[col] = dataset.df[col].astype(float)

    # Fill any remaining NaN
    remaining_nan = dataset.df.isna().sum().sum()
    if remaining_nan > 0:
        dataset.df = dataset.df.fillna(0)

    dataset.session.log(
        "prepare",
        params={"auto": auto, "final_shape": list(dataset.shape)},
    )

    return dataset


def split_data(
    dataset: DruidDataset,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split into train/test sets.

    Parameters
    ----------
    dataset : DruidDataset
        Must have ``target`` set.
    test_size : float, optional
    random_state : int, optional
    stratify : bool
        Stratify on target for classification tasks.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if dataset.target is None:
        raise ValueError("Set dataset.target before splitting")

    cfg = dataset.config.pipeline
    ts = test_size or cfg.test_size
    rs = random_state or cfg.random_state

    X = dataset.df.drop(columns=[dataset.target])
    y = dataset.df[dataset.target]

    stratify_col = y if stratify and y.nunique() <= 50 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=rs, stratify=stratify_col,
    )

    dataset.session.log(
        "split_data",
        params={
            "test_size": ts,
            "random_state": rs,
            "stratify": stratify_col is not None,
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
        },
    )

    return X_train, X_test, y_train, y_test


def build_sklearn_pipeline(
    scale: bool = True,
    pca: bool = False,
    pca_variance: float = 0.95,
) -> Pipeline:
    """
    Build a reusable sklearn preprocessing pipeline.

    Parameters
    ----------
    scale : bool
        Apply standard scaling.
    pca : bool
        Apply PCA dimensionality reduction.
    pca_variance : float
        Fraction of variance to retain if PCA is enabled.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.decomposition import PCA as SkPCA

    steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if pca:
        steps.append(("pca", SkPCA(n_components=pca_variance)))

    return Pipeline(steps)
