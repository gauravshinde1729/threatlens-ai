"""Model evaluation metrics and report generation for CVE exploit prediction."""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation.explainability import explain_prediction

logger = logging.getLogger(__name__)


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
) -> dict:
    """Compute classification metrics for a fitted model on a held-out test set.

    Args:
        model: Fitted sklearn-compatible classifier.
        X_test: Test feature DataFrame.
        y_test: True binary labels.

    Returns:
        Dict containing accuracy, precision, recall, f1_score, roc_auc,
        confusion_matrix (list[list[int]]), and classification_report (str).

        roc_auc is set to None when the test set contains only one class,
        since the metric is undefined in that case.
    """
    y_arr = np.asarray(y_test)
    y_pred = model.predict(X_test)

    # ROC-AUC requires both classes present
    roc_auc: float | None
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        logger.warning(
            "roc_auc is undefined: test set contains only class(es) %s",
            unique_classes.tolist(),
        )
        roc_auc = None
    else:
        y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        roc_auc = float(roc_auc_score(y_arr, y_score))

    cm = confusion_matrix(y_arr, y_pred)
    report = classification_report(y_arr, y_pred, zero_division=0)

    result = {
        "accuracy": float(accuracy_score(y_arr, y_pred)),
        "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_arr, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    logger.info(
        "Evaluation — acc=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%s",
        result["accuracy"],
        result["precision"],
        result["recall"],
        result["f1_score"],
        f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
    )
    return result


def generate_report(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    n_train_samples: int | None = None,
) -> dict:
    """Build a complete evaluation artifact combining metrics and SHAP explanations.

    Args:
        model: Fitted sklearn-compatible tree-based classifier.
        X_test: Test feature DataFrame.
        y_test: True binary labels.
        feature_names: Ordered list of feature column names.
        n_train_samples: Number of samples used for training (for metadata).

    Returns:
        Dict with keys:
            - metrics (dict): output of evaluate_model()
            - explainability (dict): output of explain_prediction()
            - model_metadata (dict): timestamp, n_train_samples, n_features
    """
    metrics = evaluate_model(model, X_test, y_test)
    explainability = explain_prediction(model, X_test, feature_names)

    report = {
        "metrics": metrics,
        "explainability": explainability,
        "model_metadata": {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "n_train_samples": n_train_samples,
            "n_features": len(feature_names),
        },
    }

    logger.info(
        "Report generated — %d features, n_train=%s",
        len(feature_names),
        n_train_samples,
    )
    return report
