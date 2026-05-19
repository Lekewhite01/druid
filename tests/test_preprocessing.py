"""Tests for druid.preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from druid.core.dataset import DruidDataset
from druid.preprocessing.cleaner import (
    auto_clean,
    drop_high_null_columns,
    impute_missing,
    trim_whitespace,
)
from druid.preprocessing.encoder import encode_categoricals
from druid.preprocessing.transformer import bin_numeric, treat_outliers


@pytest.fixture
def messy_dataset():
    """Create a dataset with typical data quality issues."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "amount": np.random.exponential(100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "city": np.random.choice(["Lagos", "Accra", "Nairobi", " Lagos "], n),
        "mostly_null": [np.nan] * 180 + list(range(20)),
        "constant": ["same"] * n,
        "target": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    })
    # Inject some nulls
    df.loc[0:10, "amount"] = np.nan
    df.loc[5:15, "category"] = np.nan
    return DruidDataset(df, name="messy", target="target")


class TestCleaner:
    def test_trim_whitespace(self, messy_dataset):
        ds = trim_whitespace(messy_dataset)
        # " Lagos " should be trimmed to "Lagos"
        assert " Lagos " not in ds.df["city"].values

    def test_drop_high_null(self, messy_dataset):
        ds = drop_high_null_columns(messy_dataset, threshold=0.5)
        assert "mostly_null" not in ds.df.columns

    def test_impute_missing(self, messy_dataset):
        ds = impute_missing(messy_dataset)
        # Numeric cols should have no NaN
        assert ds.df["amount"].isna().sum() == 0

    def test_auto_clean(self, messy_dataset):
        ds = auto_clean(messy_dataset)
        assert ds.df.isna().sum().sum() == 0
        assert "constant" not in ds.df.columns
        assert "mostly_null" not in ds.df.columns


class TestEncoder:
    def test_encode_categoricals(self, messy_dataset):
        ds = impute_missing(messy_dataset)
        ds = encode_categoricals(ds)
        # No object columns should remain
        assert ds.df.select_dtypes(include=["object"]).shape[1] == 0


class TestTransformer:
    def test_bin_numeric(self, messy_dataset):
        ds = bin_numeric(
            messy_dataset,
            column="amount",
            bins=[0, 50, 100, 500, 10000],
            labels=["low", "medium", "high", "very_high"],
        )
        assert "binned_amount" in ds.df.columns

    def test_treat_outliers(self, messy_dataset):
        ds = impute_missing(messy_dataset)
        original_max = ds.df["amount"].max()
        ds = treat_outliers(ds, columns=["amount"])
        # Winsorizer should cap the maximum
        assert ds.df["amount"].max() <= original_max
