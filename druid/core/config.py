"""
druid.core.config
~~~~~~~~~~~~~~~~~

Centralised configuration for DRUID sessions.

Manages API keys, default providers, display preferences, and
pipeline parameters. Configuration can be set programmatically
or loaded from environment variables / YAML files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml


# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = ("openai", "anthropic", "google")

DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
}


@dataclass
class AIConfig:
    """LLM provider settings."""

    provider: Literal["openai", "anthropic", "google"] = "openai"
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        # Resolve API key from env if not explicitly provided
        if self.api_key is None:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            self.api_key = os.getenv(env_map.get(self.provider, ""))

        # Default model per provider
        if self.model is None:
            self.model = DEFAULT_MODELS.get(self.provider, "gpt-4o")

    @property
    def is_configured(self) -> bool:
        """Return True if an API key is available."""
        return self.api_key is not None and len(self.api_key) > 0


@dataclass
class DisplayConfig:
    """Console output preferences."""

    use_rich: bool = True
    max_sample_rows: int = 5
    plot_style: str = "seaborn-v0_8-whitegrid"
    figure_dpi: int = 100
    show_code: bool = True  # Show generated code alongside results


@dataclass
class PipelineConfig:
    """Default preprocessing and modelling parameters."""

    # Missing value handling
    numeric_impute_strategy: str = "mean"
    categorical_impute_strategy: str = "most_frequent"
    high_null_threshold: float = 0.75  # Drop columns above this

    # Outliers
    outlier_method: str = "iqr"
    outlier_fold: float = 1.5

    # Encoding
    high_cardinality_threshold: int = 20
    encoding_strategy: str = "onehot"  # "onehot", "frequency", "target"

    # Train/test split
    test_size: float = 0.2
    random_state: int = 42

    # Modelling
    experiment_timeout_minutes: int = 10
    n_top_models: int = 5


@dataclass
class DruidConfig:
    """
    Top-level configuration container.

    Aggregates AI, display, and pipeline settings into a single
    object that is threaded through all DRUID operations.

    Examples
    --------
    >>> from druid.core.config import DruidConfig
    >>> cfg = DruidConfig(ai=AIConfig(provider="anthropic"))
    >>> cfg.ai.provider
    'anthropic'
    """

    ai: AIConfig = field(default_factory=AIConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    project_path: str = "./druid_output"

    # ---- Serialisation helpers ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialise config to a plain dict (excludes API keys)."""
        from dataclasses import asdict

        d = asdict(self)
        # Scrub sensitive values
        d["ai"].pop("api_key", None)
        return d

    def save(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DruidConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(
            ai=AIConfig(**data.get("ai", {})),
            display=DisplayConfig(**data.get("display", {})),
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            project_path=data.get("project_path", "./druid_output"),
        )
