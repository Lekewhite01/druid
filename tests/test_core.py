"""Tests for druid.core module."""

import pandas as pd
import pytest

from druid.core.config import AIConfig, DruidConfig, PipelineConfig
from druid.core.dataset import DruidDataset
from druid.core.session import Session


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestAIConfig:
    def test_default_provider(self):
        cfg = AIConfig()
        assert cfg.provider == "openai"

    def test_default_model_per_provider(self):
        assert AIConfig(provider="openai").model == "gpt-4o"
        assert AIConfig(provider="anthropic").model == "claude-sonnet-4-20250514"
        assert AIConfig(provider="google").model == "gemini-2.0-flash"

    def test_explicit_api_key(self):
        cfg = AIConfig(api_key="test-key")
        assert cfg.is_configured is True

    def test_no_api_key(self):
        cfg = AIConfig(api_key=None)
        # May or may not be configured depending on env vars
        assert isinstance(cfg.is_configured, bool)


class TestDruidConfig:
    def test_defaults(self):
        cfg = DruidConfig()
        assert cfg.ai.provider == "openai"
        assert cfg.pipeline.test_size == 0.2
        assert cfg.display.use_rich is True

    def test_to_dict_scrubs_api_key(self):
        cfg = DruidConfig(ai=AIConfig(api_key="secret"))
        d = cfg.to_dict()
        assert "api_key" not in d["ai"]


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(100),
        "amount": [float(x) for x in range(100)],
        "category": ["A"] * 50 + ["B"] * 50,
        "target": [0] * 90 + [1] * 10,
    })


class TestDruidDataset:
    def test_creation(self, sample_df):
        ds = DruidDataset(sample_df, name="test", target="target")
        assert ds.shape == (100, 4)
        assert ds.name == "test"
        assert ds.target == "target"

    def test_classify_columns(self, sample_df):
        ds = DruidDataset(sample_df, target="target")
        cl = ds.classify_columns()
        assert "amount" in cl["numeric"]
        assert "category" in cl["categorical"]

    def test_profile(self, sample_df):
        ds = DruidDataset(sample_df, name="test", target="target")
        profile = ds.profile()
        assert profile["shape"]["rows"] == 100
        assert profile["shape"]["columns"] == 4
        assert "amount" in profile["numeric_stats"]

    def test_profile_caching(self, sample_df):
        ds = DruidDataset(sample_df)
        p1 = ds.profile()
        p2 = ds.profile()
        assert p1 is p2  # Same object — cached

        p3 = ds.profile(force=True)
        assert p3 is not p1  # New object — forced recompute

    def test_copy(self, sample_df):
        ds = DruidDataset(sample_df, name="orig")
        ds2 = ds.copy()
        assert ds2.name == "orig"
        # Modifying copy doesn't affect original
        ds2.df["new_col"] = 1
        assert "new_col" not in ds.df.columns

    def test_session_logging(self, sample_df):
        ds = DruidDataset(sample_df, name="test")
        assert len(ds.session) == 1  # create_dataset logged
        assert ds.session.events[0].operation == "create_dataset"


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSession:
    def test_log_and_retrieve(self):
        s = Session(name="test")
        s.log("load", params={"file": "data.csv"})
        s.log("clean")
        assert len(s) == 2
        assert s.last.operation == "clean"

    def test_to_script(self):
        s = Session(name="test")
        s.log("load", code="df = pd.read_csv('data.csv')")
        s.log("clean", code="df = df.dropna()")
        script = s.to_script()
        assert "pd.read_csv" in script
        assert "dropna" in script

    def test_to_dict(self):
        s = Session(name="test")
        s.log("load")
        d = s.to_dict()
        assert d["name"] == "test"
        assert len(d["events"]) == 1
