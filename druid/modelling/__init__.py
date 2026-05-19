"""
druid.modelling
~~~~~~~~~~~~~~~

Model experimentation, evaluation, and AI-powered recommendations.
"""

from druid.modelling.evaluator import (
    classification_summary,
    feature_importance,
    regression_summary,
)
from druid.modelling.experiment import run_experiment
from druid.modelling.recommender import recommend_models

__all__ = [
    "run_experiment",
    "classification_summary",
    "regression_summary",
    "feature_importance",
    "recommend_models",
]
