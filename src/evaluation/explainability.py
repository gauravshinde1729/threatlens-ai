"""SHAP-based explainability utilities for CVE exploit prediction models."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def explain_prediction(
    model: object,
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Generate SHAP explanations for one or more predictions.

    Uses SHAP TreeExplainer on *model* (must be a tree-based estimator
    such as RandomForestClassifier, XGBClassifier, or a VotingClassifier
    whose first estimator is tree-based).

    Args:
        model: Fitted tree-based sklearn-compatible model.
        X: Input features as a DataFrame (1 or more rows).
        feature_names: Column names that correspond to model input features.

    Returns:
        Dict with keys:
            - shap_values (np.ndarray): shape (n_samples, n_features),
              SHAP values for the positive class.
            - feature_names (list[str]): feature names in column order.
            - top_3_positive (list[tuple[str, float]]): top 3 (feature, value)
              pairs with the highest mean SHAP contribution toward exploitation.
            - top_3_negative (list[tuple[str, float]]): top 3 (feature, value)
              pairs with the most negative mean SHAP contribution.
    """
    X_arr = X[feature_names] if set(feature_names).issubset(X.columns) else X

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_arr)

    # Normalise across SHAP output formats:
    #   - list [cls0, cls1]  → older scikit-learn / shap versions
    #   - 3-D array (n, f, c) → newer shap
    if isinstance(raw, list):
        values = np.array(raw[1])  # class-1 slice
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        values = raw[:, :, 1]
    else:
        values = np.array(raw)

    # Ensure 2-D even for single-row input
    if values.ndim == 1:
        values = values[np.newaxis, :]

    mean_shap = values.mean(axis=0)  # (n_features,)

    sorted_desc = np.argsort(mean_shap)[::-1]
    top_3_positive = [(feature_names[i], float(mean_shap[i])) for i in sorted_desc[:3]]

    sorted_asc = np.argsort(mean_shap)
    top_3_negative = [(feature_names[i], float(mean_shap[i])) for i in sorted_asc[:3]]

    logger.debug(
        "SHAP explain: %d samples, top+ %s, top- %s",
        len(X_arr),
        [f for f, _ in top_3_positive],
        [f for f, _ in top_3_negative],
    )

    return {
        "shap_values": values,
        "feature_names": feature_names,
        "top_3_positive": top_3_positive,
        "top_3_negative": top_3_negative,
    }


def plot_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> plt.Figure:
    """Bar chart of mean absolute SHAP values for the top-N features.

    Args:
        shap_values: Array of shape (n_samples, n_features).
        feature_names: Feature names matching the columns of *shap_values*.
        top_n: Number of top features to display (default 15).

    Returns:
        Matplotlib Figure object (caller is responsible for showing/saving).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1][:top_n]

    top_features = [feature_names[i] for i in sorted_idx]
    top_values = mean_abs[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.45)))
    ax.barh(top_features[::-1], top_values[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — Top {len(top_features)} Features")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig
