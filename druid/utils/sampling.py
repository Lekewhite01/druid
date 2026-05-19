"""
druid.utils.sampling
~~~~~~~~~~~~~~~~~~~~~

Smart data sampling for LLM context.  Instead of sending raw data
to the AI, we send compact statistical summaries that fit within
token limits while preserving the information the AI needs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def profile_for_llm(
    df: pd.DataFrame,
    target: Optional[str] = None,
    max_sample_rows: int = 5,
) -> str:
    """
    Create a compact, LLM-friendly summary of a DataFrame.

    This is optimised for token efficiency — the AI gets enough
    to reason about the data without seeing the full dataset.

    Parameters
    ----------
    df : pd.DataFrame
    target : str, optional
    max_sample_rows : int
        Number of example rows to include.

    Returns
    -------
    str
        JSON-formatted profile string.
    """
    n_rows, n_cols = df.shape

    summary: Dict[str, Any] = {
        "shape": [n_rows, n_cols],
        "columns": {},
    }

    if target and target in df.columns:
        summary["target"] = target
        t = df[target]
        if t.nunique() <= 20:
            summary["target_distribution"] = t.value_counts().to_dict()
        else:
            summary["target_stats"] = {
                "mean": float(t.mean()),
                "std": float(t.std()),
                "min": float(t.min()),
                "max": float(t.max()),
            }

    for col in df.columns:
        s = df[col]
        info: Dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_pct": round(s.isna().mean() * 100, 1),
            "nunique": int(s.nunique()),
        }

        if pd.api.types.is_numeric_dtype(s):
            clean = s.dropna()
            if len(clean) > 0:
                info["mean"] = round(float(clean.mean()), 3)
                info["std"] = round(float(clean.std()), 3)
                info["min"] = float(clean.min())
                info["max"] = float(clean.max())
        elif s.dtype == object:
            info["top_3"] = s.value_counts().head(3).to_dict()
            info["avg_len"] = round(s.dropna().astype(str).str.len().mean(), 1)

        summary["columns"][col] = info

    # Add a small sample
    sample = df.head(max_sample_rows).to_dict(orient="records")
    summary["sample_rows"] = sample

    return json.dumps(summary, indent=2, default=str)
