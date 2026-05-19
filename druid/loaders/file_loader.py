"""
druid.loaders.file_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~

Load data from local files: CSV, TSV, Parquet, Excel, JSON.
Auto-detects format from file extension.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from druid.loaders.base import BaseLoader

# Extension → reader mapping
_READERS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".feather": "feather",
}


class FileLoader(BaseLoader):
    """
    Load tabular data from local files.

    Supports CSV, TSV, Parquet, Excel (.xlsx/.xls), JSON, JSONL,
    and Feather formats.  The format is auto-detected from the
    file extension unless overridden via ``fmt``.

    Parameters
    ----------
    fmt : str, optional
        Force a specific format (e.g. ``"csv"``, ``"parquet"``).
        If omitted, the extension is used.

    Examples
    --------
    >>> loader = FileLoader()
    >>> df = loader.load("transactions.csv")
    >>> df = loader.load("data.json", orient="records")
    """

    def __init__(self, fmt: Optional[str] = None) -> None:
        self._forced_fmt = fmt

    @property
    def name(self) -> str:
        return "FileLoader"

    def can_handle(self, source: str) -> bool:
        """Return True if *source* looks like a supported local file."""
        ext = Path(source).suffix.lower()
        return ext in _READERS or self._forced_fmt is not None

    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Read a file into a DataFrame.

        Parameters
        ----------
        source : str
            File path.
        **kwargs
            Passed through to the underlying pandas reader.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the format cannot be determined.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        fmt = self._forced_fmt or _READERS.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(
                f"Unsupported file extension '{path.suffix}'. "
                f"Supported: {', '.join(_READERS.keys())}"
            )

        if fmt == "csv":
            return pd.read_csv(path, **kwargs)
        elif fmt == "tsv":
            return pd.read_csv(path, sep="\t", **kwargs)
        elif fmt == "parquet":
            return pd.read_parquet(path, **kwargs)
        elif fmt == "excel":
            return pd.read_excel(path, **kwargs)
        elif fmt == "json":
            return pd.read_json(path, **kwargs)
        elif fmt == "jsonl":
            return pd.read_json(path, lines=True, **kwargs)
        elif fmt == "feather":
            return pd.read_feather(path, **kwargs)
        else:
            raise ValueError(f"Unknown format: {fmt}")
