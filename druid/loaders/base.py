"""
druid.loaders.base
~~~~~~~~~~~~~~~~~~

Abstract base class for all DRUID data loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd


class BaseLoader(ABC):
    """
    Interface that every loader must implement.

    Subclasses handle a specific source type (file, database, API)
    and return a plain ``pd.DataFrame``.
    """

    @abstractmethod
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from *source* and return a DataFrame.

        Parameters
        ----------
        source : str
            Path, URI, or table reference.
        **kwargs
            Loader-specific options.

        Returns
        -------
        pd.DataFrame
        """
        ...

    @abstractmethod
    def can_handle(self, source: str) -> bool:
        """Return True if this loader can handle the given source."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable loader name."""
        ...
