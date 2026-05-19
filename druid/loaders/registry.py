"""
druid.loaders.registry
~~~~~~~~~~~~~~~~~~~~~~

Auto-detection registry that picks the right loader for a given
source string (file path, table reference, SQL query, etc.).
"""

from __future__ import annotations

from typing import List, Optional, Type

import pandas as pd

from druid.loaders.base import BaseLoader
from druid.loaders.file_loader import FileLoader
from druid.loaders.database_loader import BigQueryLoader, SQLLoader


# Default loader chain — checked in order
_DEFAULT_LOADERS: List[BaseLoader] = [
    FileLoader(),
    BigQueryLoader(),
]


def auto_load(
    source: str,
    loaders: Optional[List[BaseLoader]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Automatically detect the right loader and load data.

    Iterates through registered loaders and uses the first one
    that reports ``can_handle(source) == True``.

    Parameters
    ----------
    source : str
        File path, table reference, SQL query, or URI.
    loaders : list of BaseLoader, optional
        Custom loader chain.  Uses defaults if omitted.
    **kwargs
        Passed to the chosen loader's ``load()`` method.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If no loader can handle the source.

    Examples
    --------
    >>> df = auto_load("data.csv")
    >>> df = auto_load("project.dataset.table")
    >>> df = auto_load("SELECT * FROM users", loaders=[SQLLoader("sqlite:///db.sqlite")])
    """
    chain = loaders or _DEFAULT_LOADERS

    for loader in chain:
        if loader.can_handle(source):
            return loader.load(source, **kwargs)

    raise ValueError(
        f"No loader can handle source: {source!r}. "
        f"Tried: {[l.name for l in chain]}"
    )
