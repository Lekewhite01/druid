"""
druid.ai.schema_inspector
~~~~~~~~~~~~~~~~~~~~~~~~~~

AI-powered schema analysis.  Takes a dataset profile, sends it to
the configured LLM, and returns structured observations and
recommendations.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from druid.ai.provider import LLMProvider, get_provider
from druid.ai.prompts import SYSTEM_PROMPT, schema_inspection_prompt
from druid.core.config import DruidConfig
from druid.core.dataset import DruidDataset


def inspect_schema(
    dataset: DruidDataset,
    config: Optional[DruidConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """
    Run AI-powered schema inspection on a DruidDataset.

    Computes the dataset profile, sends it to the configured LLM,
    and returns the AI's analysis as structured text.

    Parameters
    ----------
    dataset : DruidDataset
        The dataset to inspect.
    config : DruidConfig, optional
        Override the dataset's config.
    provider : LLMProvider, optional
        Pre-instantiated provider.  If omitted, one is created
        from the config.

    Returns
    -------
    dict
        Keys: ``raw_response`` (str), ``profile`` (dict),
        ``provider`` (str), ``model`` (str).

    Examples
    --------
    >>> from druid.ai.schema_inspector import inspect_schema
    >>> result = inspect_schema(ds)
    >>> print(result["raw_response"])
    """
    cfg = config or dataset.config
    llm = provider or get_provider(cfg.ai)

    # Compute profile (uses cache if available)
    profile = dataset.profile()

    # Build prompt and call LLM
    user_msg = schema_inspection_prompt(profile)
    response = llm.complete(user_message=user_msg, system_prompt=SYSTEM_PROMPT)

    result = {
        "raw_response": response,
        "profile": profile,
        "provider": llm.provider_name,
        "model": cfg.ai.model,
    }

    # Log to session
    dataset.session.log(
        "inspect_schema",
        params={"provider": llm.provider_name, "model": cfg.ai.model},
        ai_rationale=response[:500],  # Truncate for session log
    )

    return result
