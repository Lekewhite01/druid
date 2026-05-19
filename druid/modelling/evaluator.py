"""
druid.modelling.evaluator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Model evaluation utilities: confusion matrix, classification report,
ROC curves, feature importance, and learning curves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def classification_summary(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full classification evaluation: report, confusion matrix, ROC curve.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : array-like
    class_names : list of str, optional

    Returns
    -------
    dict
        Keys: ``report`` (str), ``figures`` (dict of Figure).
    """
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    figures: Dict[str, plt.Figure] = {}

    # Confusion matrix
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax_cm, cmap="Blues",
        display_labels=class_names,
    )
    ax_cm.set_title("Confusion Matrix")
    figures["confusion_matrix"] = fig_cm

    # ROC curve (binary only)
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
        ax_roc.set_title("ROC Curve")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
        figures["roc_curve"] = fig_roc

    return {"report": report, "figures": figures}


def regression_summary(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Full regression evaluation: metrics, residual plot, predicted vs actual.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test, y_test : array-like

    Returns
    -------
    dict
    """
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    metrics = {
        "r2": round(r2_score(y_test, y_pred), 4),
        "mae": round(mean_absolute_error(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    }

    figures: Dict[str, plt.Figure] = {}

    # Predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", alpha=0.7)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")

    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    figures["regression_plots"] = fig

    return {"metrics": metrics, "figures": figures}


def feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    top_n: int = 20,
) -> Optional[plt.Figure]:
    """
    Plot feature importance for tree-based or linear models.

    Parameters
    ----------
    model : fitted estimator
    feature_names : list of str
    top_n : int

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    importances = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    # Linear models
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None:
        return None

    # Sort and take top_n
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(range(len(top_features)), top_values[::-1], color="#3498db")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    return fig
