"""
druid.core.dataset
~~~~~~~~~~~~~~~~~~

The ``DruidDataset`` is the central object in DRUID.  It wraps a
pandas DataFrame with rich metadata — column profiles, AI-generated
insights, transformation history, and session tracking.

Every DRUID operation (inspect, clean, visualise, train) takes and
returns a ``DruidDataset``, keeping the full workflow auditable.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from druid.core.config import DruidConfig
from druid.core.session import Session


class DruidDataset:
    """
    AI-aware wrapper around a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The underlying data.
    name : str, optional
        Human-friendly dataset label.
    target : str, optional
        Name of the target / label column (if known).
    config : DruidConfig, optional
        DRUID configuration.  Uses defaults if omitted.

    Attributes
    ----------
    df : pd.DataFrame
        The live data.
    name : str
        Dataset label.
    target : str or None
        Target column name.
    config : DruidConfig
        Active configuration.
    session : Session
        Operation history.
    metadata : dict
        Arbitrary key-value store for user annotations.

    Examples
    --------
    >>> import pandas as pd
    >>> from druid.core.dataset import DruidDataset
    >>> ds = DruidDataset(pd.read_csv("data.csv"), name="fraud", target="is_fraud")
    >>> ds.shape
    (10000, 25)
    >>> ds.profile()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: Optional[str] = None,
        target: Optional[str] = None,
        config: Optional[DruidConfig] = None,
    ) -> None:
        self.df = df.copy()
        self.name = name or "untitled"
        self.target = target
        self.config = config or DruidConfig()
        self.session = Session(name=self.name)
        self.metadata: Dict[str, Any] = {}

        # Cache for expensive computations
        self._profile_cache: Optional[Dict[str, Any]] = None
        self._ai_insights: Optional[Dict[str, Any]] = None

        # Log the creation
        self.session.log(
            "create_dataset",
            params={"name": self.name, "shape": list(self.df.shape), "target": self.target},
        )

    # ------------------------------------------------------------------
    # DataFrame pass-through properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (n_rows, n_columns)."""
        return self.df.shape

    @property
    def columns(self) -> pd.Index:
        """Return column names."""
        return self.df.columns

    @property
    def dtypes(self) -> pd.Series:
        """Return column dtypes."""
        return self.df.dtypes

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first *n* rows."""
        return self.df.head(n)

    def describe(self, **kwargs) -> pd.DataFrame:
        """Delegate to ``DataFrame.describe``."""
        return self.df.describe(**kwargs)

    # ------------------------------------------------------------------
    # Column classification
    # ------------------------------------------------------------------

    def classify_columns(self) -> Dict[str, List[str]]:
        """
        Classify columns into semantic types.

        Returns
        -------
        dict
            Keys: ``numeric``, ``categorical``, ``boolean``,
            ``datetime``, ``text``, ``id_like``.
        """
        classification: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "boolean": [],
            "datetime": [],
            "text": [],
            "id_like": [],
        }

        for col in self.df.columns:
            series = self.df[col]
            dtype = series.dtype

            # Datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                classification["datetime"].append(col)
                continue

            # Boolean
            if pd.api.types.is_bool_dtype(dtype):
                classification["boolean"].append(col)
                continue

            # Numeric
            if pd.api.types.is_numeric_dtype(dtype):
                nunique = series.nunique()
                # Binary flags
                if nunique <= 2:
                    classification["boolean"].append(col)
                # Likely an ID column (high cardinality integer, unique-ish)
                elif nunique / len(series) > 0.9 and pd.api.types.is_integer_dtype(dtype):
                    classification["id_like"].append(col)
                else:
                    classification["numeric"].append(col)
                continue

            # Object / string
            if dtype == object or pd.api.types.is_string_dtype(dtype):
                nunique = series.nunique()
                avg_len = series.dropna().astype(str).str.len().mean()

                # ID-like: very high cardinality
                if nunique / max(len(series), 1) > 0.9:
                    classification["id_like"].append(col)
                # Free text: long average string length
                elif avg_len > 50:
                    classification["text"].append(col)
                else:
                    classification["categorical"].append(col)
                continue

            # Fallback
            classification["categorical"].append(col)

        return classification

    # ------------------------------------------------------------------
    # Quick profile
    # ------------------------------------------------------------------

    def profile(self, force: bool = False) -> Dict[str, Any]:
        """
        Compute a statistical profile of the dataset.

        This is a lightweight alternative to full profiling libraries,
        designed to produce a compact summary suitable for sending to
        an LLM for analysis.

        Parameters
        ----------
        force : bool
            Recompute even if a cached profile exists.

        Returns
        -------
        dict
            Profile with keys: ``shape``, ``columns``, ``dtypes``,
            ``missing``, ``numeric_stats``, ``categorical_stats``,
            ``classification``.
        """
        if self._profile_cache is not None and not force:
            return self._profile_cache

        n_rows, n_cols = self.df.shape
        missing = self.df.isna().sum()
        classification = self.classify_columns()

        # Numeric summary
        num_cols = classification["numeric"]
        numeric_stats = {}
        if num_cols:
            desc = self.df[num_cols].describe().to_dict()
            for col in num_cols:
                numeric_stats[col] = {
                    "mean": desc[col].get("mean"),
                    "std": desc[col].get("std"),
                    "min": desc[col].get("min"),
                    "max": desc[col].get("max"),
                    "median": desc[col].get("50%"),
                    "null_pct": round(missing[col] / n_rows * 100, 2),
                    "nunique": int(self.df[col].nunique()),
                }

        # Categorical summary
        cat_cols = classification["categorical"]
        categorical_stats = {}
        for col in cat_cols:
            top_values = self.df[col].value_counts().head(5).to_dict()
            categorical_stats[col] = {
                "nunique": int(self.df[col].nunique()),
                "null_pct": round(missing[col] / n_rows * 100, 2),
                "top_values": {str(k): int(v) for k, v in top_values.items()},
            }

        profile = {
            "name": self.name,
            "shape": {"rows": n_rows, "columns": n_cols},
            "target": self.target,
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing": {
                col: {"count": int(v), "pct": round(v / n_rows * 100, 2)}
                for col, v in missing.items()
                if v > 0
            },
            "duplicates": int(self.df.duplicated().sum()),
            "classification": classification,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
        }

        self._profile_cache = profile
        return profile

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def copy(self) -> "DruidDataset":
        """Return a deep copy of this dataset."""
        new = DruidDataset(
            df=self.df.copy(),
            name=self.name,
            target=self.target,
            config=copy.deepcopy(self.config),
        )
        new.session = self.session  # Share session history
        new.metadata = dict(self.metadata)
        return new

    def __repr__(self) -> str:
        target_info = f", target={self.target!r}" if self.target else ""
        return (
            f"DruidDataset(name={self.name!r}, "
            f"rows={self.shape[0]:,}, cols={self.shape[1]}"
            f"{target_info})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _repr_html_(self) -> str:
        """Rich HTML rendering in Jupyter notebooks."""
        return (
            f"<b>DruidDataset</b>: {self.name} "
            f"({self.shape[0]:,} rows × {self.shape[1]} cols)"
            f"{'  |  target: ' + self.target if self.target else ''}"
            f"<br/>{self.df.head(3)._repr_html_()}"
        )
