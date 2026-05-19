"""
druid.modelling.experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AutoML-style experiment runner.  Trains multiple algorithms on the
prepared dataset, evaluates each, and returns a ranked leaderboard.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from druid.core.dataset import DruidDataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

CLASSIFIERS: Dict[str, BaseEstimator] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_jobs=-1),
    "SGD": SGDClassifier(random_state=42, max_iter=1000),
}

REGRESSORS: Dict[str, BaseEstimator] = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_jobs=-1),
}

# Optional models — loaded if available
_OPTIONAL_CLASSIFIERS: Dict[str, str] = {
    "XGBoost": "xgboost.XGBClassifier",
    "LightGBM": "lightgbm.LGBMClassifier",
}
_OPTIONAL_REGRESSORS: Dict[str, str] = {
    "XGBoost": "xgboost.XGBRegressor",
    "LightGBM": "lightgbm.LGBMRegressor",
}


def _load_optional_models(
    registry: Dict[str, BaseEstimator],
    optional: Dict[str, str],
) -> Dict[str, BaseEstimator]:
    """Try to import optional models and add them to the registry."""
    import importlib

    for name, class_path in optional.items():
        module_path, class_name = class_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            kwargs = {"random_state": 42, "n_jobs": -1, "verbosity": 0}
            # Filter kwargs to only what the class accepts
            import inspect
            valid = inspect.signature(cls.__init__).parameters
            kwargs = {k: v for k, v in kwargs.items() if k in valid}
            registry[name] = cls(**kwargs)
        except (ImportError, AttributeError):
            pass  # Optional dependency not installed

    return registry


def _evaluate_classifier(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # AUC — only for binary and when predict_proba is available
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    return {k: round(v, 4) for k, v in metrics.items()}


def _evaluate_regressor(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute regression metrics."""
    y_pred = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    }


def run_experiment(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    y_test: pd.Series | np.ndarray,
    task_type: str = "classification",
    models: Optional[Dict[str, BaseEstimator]] = None,
    sort_by: Optional[str] = None,
    dataset: Optional[DruidDataset] = None,
) -> Dict[str, Any]:
    """
    Train multiple models and return a ranked leaderboard.

    Parameters
    ----------
    X_train, X_test : array-like
        Feature matrices.
    y_train, y_test : array-like
        Target vectors.
    task_type : str
        ``"classification"`` or ``"regression"``.
    models : dict, optional
        Custom model registry.  Uses defaults if omitted.
    sort_by : str, optional
        Metric to rank by.  Defaults to ``"f1"`` for classification,
        ``"r2"`` for regression.
    dataset : DruidDataset, optional
        For session logging.

    Returns
    -------
    dict
        Keys: ``leaderboard`` (list of dicts), ``task_type``,
        ``best_model_name``, ``best_model``, ``all_models``.
    """
    if task_type == "classification":
        registry = dict(CLASSIFIERS)
        _load_optional_models(registry, _OPTIONAL_CLASSIFIERS)
        evaluate = _evaluate_classifier
        sort_by = sort_by or "f1"
    else:
        registry = dict(REGRESSORS)
        _load_optional_models(registry, _OPTIONAL_REGRESSORS)
        evaluate = _evaluate_regressor
        sort_by = sort_by or "r2"

    if models is not None:
        registry = models

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, BaseEstimator] = {}

    for name, model in registry.items():
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)
            metrics = evaluate(model, X_test, y_test)
            metrics["train_time_sec"] = elapsed
            metrics["model"] = name
            results.append(metrics)
            trained_models[name] = model
        except Exception as e:
            results.append({
                "model": name,
                "error": str(e),
                "train_time_sec": round(time.time() - t0, 2),
            })

    # Sort leaderboard
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    successful.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    leaderboard = successful + failed

    best_name = leaderboard[0]["model"] if successful else None

    result = {
        "task_type": task_type,
        "leaderboard": leaderboard,
        "best_model_name": best_name,
        "best_model": trained_models.get(best_name),
        "all_models": trained_models,
        "sort_by": sort_by,
    }

    # Log to session
    if dataset:
        dataset.session.log(
            "run_experiment",
            params={
                "task_type": task_type,
                "n_models": len(registry),
                "best_model": best_name,
                "best_score": leaderboard[0].get(sort_by) if successful else None,
            },
        )

    return result
