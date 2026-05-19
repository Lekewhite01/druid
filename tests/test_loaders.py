"""Tests for druid.loaders module."""

import os
import tempfile

import pandas as pd
import pytest

from druid.loaders.file_loader import FileLoader
from druid.loaders.database_loader import BigQueryLoader
from druid.loaders.registry import auto_load


@pytest.fixture
def csv_file():
    """Create a temp CSV file."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def parquet_file():
    """Create a temp Parquet file."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        df.to_parquet(f.name)
        yield f.name
    os.unlink(f.name)


class TestFileLoader:
    def test_can_handle_csv(self):
        assert FileLoader().can_handle("data.csv") is True

    def test_can_handle_parquet(self):
        assert FileLoader().can_handle("data.parquet") is True

    def test_cannot_handle_unknown(self):
        assert FileLoader().can_handle("data.xyz") is False

    def test_load_csv(self, csv_file):
        df = FileLoader().load(csv_file)
        assert len(df) == 3
        assert list(df.columns) == ["a", "b"]

    def test_load_parquet(self, parquet_file):
        df = FileLoader().load(parquet_file)
        assert len(df) == 3

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            FileLoader().load("/nonexistent/file.csv")


class TestBigQueryLoader:
    def test_can_handle_table_ref(self):
        assert BigQueryLoader().can_handle("project.dataset.table") is True
        assert BigQueryLoader().can_handle("dataset.table") is True

    def test_can_handle_sql(self):
        assert BigQueryLoader().can_handle("SELECT * FROM table") is True

    def test_cannot_handle_file(self):
        assert BigQueryLoader().can_handle("data.csv") is False


class TestAutoLoad:
    def test_auto_load_csv(self, csv_file):
        df = auto_load(csv_file)
        assert len(df) == 3

    def test_auto_load_unknown(self):
        with pytest.raises(ValueError, match="No loader"):
            auto_load("unknown_source_format")
