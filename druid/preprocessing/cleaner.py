"""
druid.preprocessing.cleaner
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data cleaning operations: handle missing values, drop uninformative
columns, trim whitespace, fix dtypes, and remove duplicates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropConstantFeatures

from druid.core.dataset import DruidDataset


def drop_high_null_columns(
    dataset: DruidDataset,
    threshold: Optional[float] = None,
    exclude: Optional[List[str]] = None,
) -> DruidDataset:
    """
    Drop columns where the missing-value percentage exceeds *threshold*.

    Parameters
    ----------
    dataset : DruidDataset
    threshold : float, optional
        Fraction (0–1) above which a column is dropped.
        Defaults to ``config.pipeline.high_null_threshold``.
    exclude : list of str, optional
        Column names to never drop (e.g. the target).

    Returns
    -------
    DruidDataset
        Updated dataset (modifies in place and returns self).
    """
    thresh = threshold or dataset.config.pipeline.high_null_threshold
    exclude = set(exclude or [])
    if dataset.target:
        exclude.add(dataset.target)

    null_pct = dataset.df.isna().mean()
    to_drop = [
        col for col, pct in null_pct.items()
        if pct > thresh and col not in exclude
    ]

    if to_drop:
        dataset.df.drop(columns=to_drop, inplace=True)
        dataset.session.log(
            "drop_high_null_columns",
            params={"threshold": thresh, "dropped": to_drop},
            code=f"df.drop(columns={to_drop}, inplace=True)",
        )

    return dataset


def drop_constant_features(
    dataset: DruidDataset,
    tol: float = 0.98,
) -> DruidDataset:
    """
    Drop constant and quasi-constant columns using feature-engine.

    Parameters
    ----------
    dataset : DruidDataset
    tol : float
        Tolerance — drop columns where a single value represents
        >= *tol* fraction of rows.

    Returns
    -------
    DruidDataset
    """
    pre_cols = set(dataset.df.columns)

    dcf = DropConstantFeatures(tol=tol, missing_values="include")
    dataset.df = dcf.fit_transform(dataset.df)

    dropped = sorted(pre_cols - set(dataset.df.columns))
    if dropped:
        dataset.session.log(
            "drop_constant_features",
            params={"tol": tol, "dropped": dropped},
        )

    return dataset


def impute_missing(
    dataset: DruidDataset,
    numeric_strategy: Optional[str] = None,
    categorical_strategy: Optional[str] = None,
) -> DruidDataset:
    """
    Impute missing values using feature-engine.

    Parameters
    ----------
    dataset : DruidDataset
    numeric_strategy : str, optional
        ``"mean"`` or ``"median"``.  Defaults to config.
    categorical_strategy : str, optional
        ``"frequent"`` or ``"missing"`` (fill with string 'Missing').

    Returns
    -------
    DruidDataset
    """
    cfg = dataset.config.pipeline
    num_strat = numeric_strategy or cfg.numeric_impute_strategy
    cat_strat = categorical_strategy or cfg.categorical_impute_strategy

    target = dataset.target or ""

    # Numeric imputation
    num_cols = [
        c for c in dataset.df.select_dtypes(include=[np.number]).columns
        if dataset.df[c].isna().any() and c != target
    ]
    if num_cols:
        imputer = MeanMedianImputer(
            imputation_method=num_strat,
            variables=num_cols,
        )
        dataset.df = imputer.fit_transform(dataset.df)

    # Categorical imputation
    cat_cols = [
        c for c in dataset.df.select_dtypes(include=["object", "category"]).columns
        if dataset.df[c].isna().any()
    ]
    if cat_cols:
        fe_strategy = "frequent" if cat_strat == "most_frequent" else cat_strat
        imputer = CategoricalImputer(
            imputation_method=fe_strategy,
            variables=cat_cols,
        )
        dataset.df = imputer.fit_transform(dataset.df)

    dataset.session.log(
        "impute_missing",
        params={
            "numeric_strategy": num_strat,
            "numeric_cols": num_cols,
            "categorical_strategy": cat_strat,
            "categorical_cols": cat_cols,
        },
    )

    return dataset


def trim_whitespace(dataset: DruidDataset) -> DruidDataset:
    """Strip leading/trailing whitespace from all string columns."""
    str_cols = dataset.df.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        dataset.df[col] = dataset.df[col].str.strip()

    dataset.session.log("trim_whitespace", params={"columns": str_cols})
    return dataset


def remove_duplicates(
    dataset: DruidDataset,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> DruidDataset:
    """
    Remove duplicate rows.

    Parameters
    ----------
    dataset : DruidDataset
    subset : list of str, optional
        Columns to consider for duplicates.  All columns if omitted.
    keep : str
        Which duplicate to keep: ``"first"``, ``"last"``, or ``False``.

    Returns
    -------
    DruidDataset
    """
    before = len(dataset.df)
    dataset.df = dataset.df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(dataset.df)

    dataset.session.log(
        "remove_duplicates",
        params={"subset": subset, "keep": keep, "removed": removed},
    )

    return dataset


def auto_clean(dataset: DruidDataset) -> DruidDataset:
    """
    Run the full automatic cleaning pipeline:

    1. Trim whitespace
    2. Remove exact duplicates
    3. Drop columns with >75% missing
    4. Drop constant / quasi-constant columns
    5. Impute remaining missing values

    Parameters
    ----------
    dataset : DruidDataset

    Returns
    -------
    DruidDataset
    """
    dataset = trim_whitespace(dataset)
    dataset = remove_duplicates(dataset)
    dataset = drop_high_null_columns(dataset)
    dataset = drop_constant_features(dataset)
    dataset = impute_missing(dataset)

    dataset.session.log("auto_clean", params={"final_shape": list(dataset.shape)})
    return dataset
