"""
druid.ai.prompts
~~~~~~~~~~~~~~~~

Prompt templates for every AI-assisted stage of the DRUID pipeline.

Each template is a function that accepts structured data (profile dicts,
column lists, etc.) and returns a formatted prompt string.  This keeps
prompt engineering centralised and testable.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are DRUID, an expert data science assistant embedded in a Python library.
You analyse dataset profiles, suggest preprocessing steps, recommend
visualisations, and guide model selection.

Rules:
- Be concise and actionable.  Prioritise the most impactful observations.
- When suggesting code, use pandas, scikit-learn, feature-engine, and
  matplotlib/seaborn — the libraries already available in the user's env.
- Format recommendations as numbered lists.
- Flag potential data quality issues (leakage, class imbalance, encoding
  errors, suspicious distributions) prominently.
- Never hallucinate column names or statistics — use only what is provided
  in the profile.
"""


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------

def schema_inspection_prompt(profile: Dict[str, Any]) -> str:
    """
    Build a prompt asking the AI to analyse a dataset profile.

    Parameters
    ----------
    profile : dict
        Output of ``DruidDataset.profile()``.

    Returns
    -------
    str
        Formatted user prompt.
    """
    return f"""\
Analyse this dataset profile and provide:

1. **Data quality issues** — missing values, suspicious distributions,
   potential encoding errors, columns that might be mislabelled.
2. **Column type corrections** — any columns classified incorrectly
   (e.g. an ID stored as numeric, a categorical stored as float).
3. **Target variable assessment** — if a target is specified, assess
   class balance, potential leakage columns, and suitability for
   common ML tasks.
4. **Recommended next steps** — ordered by priority.

DATASET PROFILE:
```json
{json.dumps(profile, indent=2, default=str)}
```"""


# ---------------------------------------------------------------------------
# EDA guidance
# ---------------------------------------------------------------------------

def eda_guidance_prompt(
    profile: Dict[str, Any],
    user_question: Optional[str] = None,
) -> str:
    """
    Build a prompt asking the AI to recommend EDA steps.

    Parameters
    ----------
    profile : dict
        Output of ``DruidDataset.profile()``.
    user_question : str, optional
        Specific question the user wants answered.

    Returns
    -------
    str
    """
    base = f"""\
Given this dataset profile, recommend the most informative exploratory
analyses and visualisations.  For each recommendation:

- Explain *why* it matters for this specific dataset.
- Provide the Python code (pandas + matplotlib/seaborn) to produce it.
- Flag anything unusual you notice in the data.

DATASET PROFILE:
```json
{json.dumps(profile, indent=2, default=str)}
```"""

    if user_question:
        base += f"\n\nThe user specifically asks: {user_question}"

    return base


# ---------------------------------------------------------------------------
# Preprocessing recommendations
# ---------------------------------------------------------------------------

def preprocessing_prompt(
    profile: Dict[str, Any],
    target: Optional[str] = None,
) -> str:
    """
    Build a prompt asking the AI to recommend preprocessing steps.

    Parameters
    ----------
    profile : dict
        Dataset profile.
    target : str, optional
        Target column name.

    Returns
    -------
    str
    """
    return f"""\
Recommend a preprocessing pipeline for this dataset.  For each step:

1. What to do (e.g. impute, encode, scale, drop, engineer).
2. Which columns it applies to.
3. Why (brief rationale).
4. The Python code using pandas / scikit-learn / feature-engine.

Consider: missing values, outliers, categorical encoding, datetime
features, class imbalance, feature selection, and any data quality
issues visible in the profile.

Target column: {target or 'not specified'}

DATASET PROFILE:
```json
{json.dumps(profile, indent=2, default=str)}
```"""


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def model_recommendation_prompt(
    profile: Dict[str, Any],
    task_type: str,
    experiment_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a prompt asking the AI to recommend models.

    Parameters
    ----------
    profile : dict
        Dataset profile.
    task_type : str
        ``"classification"`` or ``"regression"``.
    experiment_results : dict, optional
        Results from a prior experiment run.

    Returns
    -------
    str
    """
    prompt = f"""\
Recommend the best machine learning approach for this dataset.

Task type: {task_type}

Consider:
- Dataset size and dimensionality
- Class balance (if classification)
- Feature types (numeric vs categorical mix)
- Likely non-linearities

For each recommended model:
1. Why it suits this data
2. Key hyperparameters to tune
3. Expected strengths and weaknesses

DATASET PROFILE:
```json
{json.dumps(profile, indent=2, default=str)}
```"""

    if experiment_results:
        prompt += f"""

EXPERIMENT RESULTS (prior run):
```json
{json.dumps(experiment_results, indent=2, default=str)}
```

Based on these results, which model should the user invest in tuning,
and what specific hyperparameter adjustments do you recommend?"""

    return prompt


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------

def insight_prompt(
    profile: Dict[str, Any],
    analysis_results: Dict[str, Any],
) -> str:
    """
    Build a prompt asking the AI to generate natural-language insights.

    Parameters
    ----------
    profile : dict
        Dataset profile.
    analysis_results : dict
        Results from EDA (correlations, distributions, etc.).

    Returns
    -------
    str
    """
    return f"""\
Generate clear, non-technical insights from this analysis.
Write as if briefing a product manager — focus on what the data
*means*, not the statistical method used.

DATASET PROFILE:
```json
{json.dumps(profile, indent=2, default=str)}
```

ANALYSIS RESULTS:
```json
{json.dumps(analysis_results, indent=2, default=str)}
```"""
