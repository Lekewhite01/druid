"""
druid.preprocessing
~~~~~~~~~~~~~~~~~~~

Data cleaning, feature engineering, encoding, and pipeline building.
"""

from druid.preprocessing.cleaner import (
    auto_clean,
    drop_constant_features,
    drop_high_null_columns,
    impute_missing,
    remove_duplicates,
    trim_whitespace,
)
from druid.preprocessing.encoder import encode_categoricals, target_encode
from druid.preprocessing.pipeline import build_sklearn_pipeline, prepare, split_data
from druid.preprocessing.transformer import (
    bin_numeric,
    featurize_datetime,
    treat_outliers,
)

__all__ = [
    "auto_clean",
    "drop_constant_features",
    "drop_high_null_columns",
    "impute_missing",
    "remove_duplicates",
    "trim_whitespace",
    "encode_categoricals",
    "target_encode",
    "build_sklearn_pipeline",
    "prepare",
    "split_data",
    "bin_numeric",
    "featurize_datetime",
    "treat_outliers",
]
