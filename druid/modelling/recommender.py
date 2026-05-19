"""
druid.modelling.recommender
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI-powered model recommendations based on dataset profile
and experiment results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from druid.ai.prompts import SYSTEM_PROMPT, model_recommendation_prompt
from druid.ai.provider import LLMProvider, get_provider
from druid.core.config import DruidConfig
from druid.core.dataset import DruidDataset


def recommend_models(
    dataset: DruidDataset,
    task_type: str = "classification",
    experiment_results: Optional[Dict[str, Any]] = None,
    config: Optional[DruidConfig] = None,
    provider: Optional[LLMProvider] = None,
) -> str:
    """
    Get AI-powered model recommendations.

    Parameters
    ----------
    dataset : DruidDataset
    task_type : str
        ``"classification"`` or ``"regression"``.
    experiment_results : dict, optional
        Results from a prior ``run_experiment()`` call.
    config : DruidConfig, optional
    provider : LLMProvider, optional

    Returns
    -------
    str
        AI-generated recommendations.
    """
    cfg = config or dataset.config
    llm = provider or get_provider(cfg.ai)

    profile = dataset.profile()
    prompt = model_recommendation_prompt(profile, task_type, experiment_results)
    response = llm.complete(user_message=prompt, system_prompt=SYSTEM_PROMPT)

    dataset.session.log(
        "recommend_models",
        params={"task_type": task_type, "has_prior_results": experiment_results is not None},
        ai_rationale=response[:500],
    )

    return response
