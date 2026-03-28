"""Tests for src/evaluation/explainability.py and src/evaluation/metrics.py."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display/Tk required

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from evaluation.explainability import explain_prediction, plot_feature_importance
from evaluation.metrics import evaluate_model, generate_report

# ---------------------------------------------------------------------------
# Shared synthetic dataset (module-scoped for speed)
# ---------------------------------------------------------------------------

_N_SAMPLES = 200
_N_FEATURES = 10
_FEATURE_NAMES = [f"feature_{i}" for i in range(_N_FEATURES)]


@pytest.fixture(scope="module")
def dataset():
    X_raw, y_raw = make_classification(
        n_samples=_N_SAMPLES,
        n_features=_N_FEATURES,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    X = pd.DataFrame(X_raw, columns=_FEATURE_NAMES)
    y = pd.Series(y_raw)
    return X, y


@pytest.fixture(scope="module")
def fitted_rf(dataset):
    X, y = dataset
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    return rf


@pytest.fixture(scope="module")
def shap_result(fitted_rf, dataset):
    X, _ = dataset
    return explain_prediction(fitted_rf, X, _FEATURE_NAMES)


# ---------------------------------------------------------------------------
# test_explain_prediction_returns_correct_structure
# ---------------------------------------------------------------------------


def test_explain_prediction_returns_correct_structure(shap_result):
    """explain_prediction() returns a dict with all required keys."""
    assert set(shap_result.keys()) == {
        "shap_values",
        "feature_names",
        "top_3_positive",
        "top_3_negative",
    }


def test_explain_prediction_shap_values_shape(shap_result, dataset):
    """shap_values has shape (n_samples, n_features)."""
    X, _ = dataset
    assert isinstance(shap_result["shap_values"], np.ndarray)
    assert shap_result["shap_values"].shape == (len(X), _N_FEATURES)


def test_explain_prediction_feature_names_match(shap_result):
    """feature_names in result matches the list passed in."""
    assert shap_result["feature_names"] == _FEATURE_NAMES


def test_explain_prediction_single_row(fitted_rf, dataset):
    """explain_prediction() works on a single-row DataFrame."""
    X, _ = dataset
    result = explain_prediction(fitted_rf, X.iloc[:1], _FEATURE_NAMES)

    assert result["shap_values"].shape == (1, _N_FEATURES)
    assert len(result["top_3_positive"]) == 3
    assert len(result["top_3_negative"]) == 3


# ---------------------------------------------------------------------------
# test_top_features_are_sorted_by_importance
# ---------------------------------------------------------------------------


def test_top_3_positive_are_tuples_of_name_and_value(shap_result):
    """top_3_positive entries are (str, float) tuples."""
    for name, value in shap_result["top_3_positive"]:
        assert isinstance(name, str)
        assert isinstance(value, float)
        assert name in _FEATURE_NAMES


def test_top_3_negative_are_tuples_of_name_and_value(shap_result):
    """top_3_negative entries are (str, float) tuples."""
    for name, value in shap_result["top_3_negative"]:
        assert isinstance(name, str)
        assert isinstance(value, float)
        assert name in _FEATURE_NAMES


def test_top_positive_sorted_descending(shap_result):
    """top_3_positive SHAP values are in descending order."""
    values = [v for _, v in shap_result["top_3_positive"]]
    assert values == sorted(values, reverse=True)


def test_top_negative_sorted_ascending(shap_result):
    """top_3_negative SHAP values are in ascending order (most negative first)."""
    values = [v for _, v in shap_result["top_3_negative"]]
    assert values == sorted(values)


def test_top_positive_values_exceed_top_negative(shap_result):
    """The least positive feature in top_3_positive outranks top_3_negative."""
    min_positive = min(v for _, v in shap_result["top_3_positive"])
    max_negative = max(v for _, v in shap_result["top_3_negative"])
    assert min_positive >= max_negative


def test_plot_feature_importance_returns_figure(shap_result):
    """plot_feature_importance() returns a Matplotlib Figure without calling show()."""
    import matplotlib.pyplot as plt

    fig = plot_feature_importance(shap_result["shap_values"], _FEATURE_NAMES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance_top_n(shap_result):
    """plot_feature_importance() respects the top_n parameter."""
    import matplotlib.pyplot as plt

    fig = plot_feature_importance(shap_result["shap_values"], _FEATURE_NAMES, top_n=5)
    ax = fig.axes[0]
    # 5 bars on the horizontal bar chart
    assert len(ax.patches) == 5
    plt.close(fig)


# ---------------------------------------------------------------------------
# test_evaluate_model_returns_all_metrics
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_all_metrics(fitted_rf, dataset):
    """evaluate_model() returns a dict with all required metric keys."""
    X, y = dataset
    result = evaluate_model(fitted_rf, X, y)

    assert set(result.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    }


def test_evaluate_model_metric_ranges(fitted_rf, dataset):
    """All scalar metrics are in [0, 1]."""
    X, y = dataset
    result = evaluate_model(fitted_rf, X, y)

    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_evaluate_model_confusion_matrix_shape(fitted_rf, dataset):
    """confusion_matrix is a 2x2 list of lists for binary classification."""
    X, y = dataset
    result = evaluate_model(fitted_rf, X, y)

    cm = result["confusion_matrix"]
    assert len(cm) == 2
    assert all(len(row) == 2 for row in cm)
    # All elements are non-negative integers
    for row in cm:
        for val in row:
            assert isinstance(val, int) and val >= 0


def test_evaluate_model_confusion_matrix_totals(fitted_rf, dataset):
    """Sum of confusion matrix equals the number of test samples."""
    X, y = dataset
    result = evaluate_model(fitted_rf, X, y)

    total = sum(val for row in result["confusion_matrix"] for val in row)
    assert total == len(y)


def test_evaluate_model_classification_report_is_string(fitted_rf, dataset):
    """classification_report is a non-empty string."""
    X, y = dataset
    result = evaluate_model(fitted_rf, X, y)

    assert isinstance(result["classification_report"], str)
    assert len(result["classification_report"]) > 0


def test_evaluate_model_handles_single_class():
    """evaluate_model() sets roc_auc=None when test set has only one class."""
    X_raw, _ = make_classification(n_samples=50, n_features=5, random_state=0)
    X = pd.DataFrame(X_raw, columns=[f"f{i}" for i in range(5)])
    y_all_zeros = pd.Series(np.zeros(50, dtype=int))

    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf.fit(X, pd.Series(np.random.default_rng(0).integers(0, 2, 50)))

    result = evaluate_model(rf, X, y_all_zeros)
    assert result["roc_auc"] is None


# ---------------------------------------------------------------------------
# test_generate_report_combines_metrics_and_shap
# ---------------------------------------------------------------------------


def test_generate_report_combines_metrics_and_shap(fitted_rf, dataset):
    """generate_report() returns all three top-level keys."""
    X, y = dataset
    report = generate_report(fitted_rf, X, y, _FEATURE_NAMES, n_train_samples=150)

    assert set(report.keys()) == {"metrics", "explainability", "model_metadata"}


def test_generate_report_metrics_content(fitted_rf, dataset):
    """generate_report() metrics section mirrors evaluate_model() output."""
    X, y = dataset
    report = generate_report(fitted_rf, X, y, _FEATURE_NAMES)
    metrics = report["metrics"]

    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics


def test_generate_report_explainability_content(fitted_rf, dataset):
    """generate_report() explainability section has SHAP values and top features."""
    X, y = dataset
    report = generate_report(fitted_rf, X, y, _FEATURE_NAMES)
    expl = report["explainability"]

    assert "shap_values" in expl
    assert "top_3_positive" in expl
    assert "top_3_negative" in expl


def test_generate_report_model_metadata(fitted_rf, dataset):
    """generate_report() model_metadata contains timestamp, n_train_samples, n_features."""
    X, y = dataset
    report = generate_report(fitted_rf, X, y, _FEATURE_NAMES, n_train_samples=160)
    meta = report["model_metadata"]

    assert "timestamp" in meta
    assert "n_train_samples" in meta
    assert "n_features" in meta
    assert meta["n_train_samples"] == 160
    assert meta["n_features"] == _N_FEATURES
    # Timestamp should be a valid ISO 8601 string
    from datetime import datetime

    datetime.fromisoformat(meta["timestamp"])  # raises if invalid


def test_generate_report_metadata_n_train_samples_none(fitted_rf, dataset):
    """n_train_samples defaults to None when not supplied."""
    X, y = dataset
    report = generate_report(fitted_rf, X, y, _FEATURE_NAMES)
    assert report["model_metadata"]["n_train_samples"] is None
