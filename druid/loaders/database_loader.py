"""
druid.loaders.database_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load data from databases: BigQuery, PostgreSQL, MySQL, SQLite.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from druid.loaders.base import BaseLoader


class BigQueryLoader(BaseLoader):
    """
    Load data from Google BigQuery.

    Requires ``google-cloud-bigquery``.  Install via::

        pip install druid-ai[bigquery]

    Parameters
    ----------
    project : str, optional
        GCP project ID.  If omitted, uses the default project
        from your environment.

    Examples
    --------
    >>> loader = BigQueryLoader(project="my-project")
    >>> df = loader.load("dataset.table_name")
    >>> df = loader.load("SELECT * FROM `proj.dataset.table` WHERE date > '2024-01-01'")
    """

    def __init__(self, project: Optional[str] = None) -> None:
        self._project = project

    @property
    def name(self) -> str:
        return "BigQueryLoader"

    def can_handle(self, source: str) -> bool:
        """Return True if *source* looks like a BQ reference or SQL query."""
        s = source.strip().lower()
        # SQL query
        if s.startswith("select ") or s.startswith("with "):
            return True
        # project.dataset.table or dataset.table pattern
        if re.match(r"^[\w-]+\.[\w-]+\.[\w-]+$", source.strip()):
            return True
        if re.match(r"^[\w-]+\.[\w-]+$", source.strip()):
            return True
        return False

    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load from BigQuery.

        Parameters
        ----------
        source : str
            Either a fully-qualified table reference
            (``project.dataset.table`` or ``dataset.table``)
            or a SQL query string.
        **kwargs
            Passed to ``client.query().to_dataframe()``.

        Returns
        -------
        pd.DataFrame
        """
        try:
            from google.cloud import bigquery
        except ImportError:
            raise ImportError(
                "BigQuery support requires google-cloud-bigquery. "
                "Install with: pip install druid-ai[bigquery]"
            )

        client = bigquery.Client(project=self._project)

        # Determine if source is a query or table reference
        s = source.strip()
        if s.lower().startswith("select ") or s.lower().startswith("with "):
            sql = s
        else:
            # Table reference — wrap in backticks
            table_ref = s if "." in s else f"{self._project}.{s}"
            sql = f"SELECT * FROM `{table_ref}`"

        return client.query(sql).to_dataframe(**kwargs)


class SQLLoader(BaseLoader):
    """
    Load data from SQL databases via SQLAlchemy.

    Supports PostgreSQL, MySQL, SQLite, and any database with
    a SQLAlchemy-compatible driver.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy connection URI, e.g.
        ``"postgresql://user:pass@host:5432/dbname"``

    Examples
    --------
    >>> loader = SQLLoader("postgresql://user:pass@localhost/mydb")
    >>> df = loader.load("SELECT * FROM transactions WHERE amount > 1000")
    """

    def __init__(self, connection_string: str) -> None:
        self._conn_str = connection_string

    @property
    def name(self) -> str:
        return "SQLLoader"

    def can_handle(self, source: str) -> bool:
        """Return True if *source* looks like a SQL query."""
        s = source.strip().lower()
        return s.startswith("select ") or s.startswith("with ")

    def load(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Execute a SQL query and return results.

        Parameters
        ----------
        source : str
            SQL query string.
        **kwargs
            Passed to ``pd.read_sql()``.

        Returns
        -------
        pd.DataFrame
        """
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError(
                "SQL database support requires sqlalchemy. "
                "Install with: pip install druid-ai[postgres]"
            )

        engine = create_engine(self._conn_str)
        return pd.read_sql(source, engine, **kwargs)
