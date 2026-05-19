"""
druid.eda.insights
~~~~~~~~~~~~~~~~~~

AI-generated insights from EDA results.  Takes profiling output
and produces human-readable findings using the configured LLM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from druid.ai.prompts import SYSTEM_PROMPT, eda_guidance_prompt, insight_prompt
from druid.ai.provider import LLMProvider, get_provider
from druid.core.config import DruidConfig
from druid.core.dataset import DruidDataset
from druid.eda.profiler import full_profile


def generate_eda_guidance(
    dataset: DruidDataset,
    question: Optional[str] = None,
    config: Optional[DruidConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> str:
    """
    Ask the AI to recommend EDA steps for the dataset.

    Parameters
    ----------
    dataset : DruidDataset
    question : str, optional
        Specific question the user wants answered.
    config : DruidConfig, optional
    provider : LLMProvider, optional

    Returns
    -------
    str
        AI-generated EDA guidance.
    """
    cfg = config or dataset.config
    llm = provider or get_provider(cfg.ai)

    profile = dataset.profile()
    prompt = eda_guidance_prompt(profile, user_question=question)
    response = llm.complete(user_message=prompt, system_prompt=SYSTEM_PROMPT)

    dataset.session.log(
        "eda_guidance",
        params={"question": question},
        ai_rationale=response[:500],
    )

    return response


def generate_insights(
    dataset: DruidDataset,
    analysis_results: Dict[str, Any],
    config: Optional[DruidConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> str:
    """
    Generate natural-language insights from analysis results.

    Parameters
    ----------
    dataset : DruidDataset
    analysis_results : dict
        Output from profiling or correlation analysis.
    config : DruidConfig, optional
    provider : LLMProvider, optional

    Returns
    -------
    str
        Human-readable insights.
    """
    cfg = config or dataset.config
    llm = provider or get_provider(cfg.ai)

    profile = dataset.profile()
    prompt = insight_prompt(profile, analysis_results)
    response = llm.complete(user_message=prompt, system_prompt=SYSTEM_PROMPT)

    dataset.session.log(
        "generate_insights",
        ai_rationale=response[:500],
    )

    return response
