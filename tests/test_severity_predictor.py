"""Tests for src/models/severity_predictor.py and src/models/model_registry.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from data.preprocessor import TOP_CWES, _KEYWORD_PATTERNS
from models.model_registry import ModelRegistry
from models.severity_predictor import SeverityPredictor

# ---------------------------------------------------------------------------
# Feature column list (mirrors preprocessor._REQUIRED_COLUMNS minus id/target)
# ---------------------------------------------------------------------------

_FEATURE_COLS: list[str] = (
    [
        "cvss_v3_score",
        "description_length",
        "reference_count",
        "affected_product_count",
        "days_since_publication",
        "attack_vector",
        "attack_complexity",
        "privileges_required",
        "user_interaction",
        "scope",
        "confidentiality_impact",
        "integrity_impact",
        "availability_impact",
    ]
    + list(_KEYWORD_PATTERNS.keys())
    + ["has_exploit_ref"]
    + [f"cwe_{cwe.split('-')[1].lower()}" for cwe in TOP_CWES]
    + ["cwe_other"]
)
_N_FEATURES = len(_FEATURE_COLS)  # 30


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """200-sample synthetic dataset with columns matching the preprocessor output."""
    X_raw, y_raw = make_classification(
        n_samples=200,
        n_features=_N_FEATURES,
        n_informative=12,
        n_redundant=4,
        random_state=42,
        flip_y=0.05,
    )
    X = pd.DataFrame(X_raw, columns=_FEATURE_COLS)
    # Clip boolean/ordinal columns to sane ranges
    for col in _FEATURE_COLS:
        if col.startswith(("has_", "cwe_")):
            X[col] = (X[col] > 0).astype(int)
    y = pd.Series(y_raw, name="is_exploited")
    return X, y


@pytest.fixture()
def fitted_predictor(synthetic_data) -> SeverityPredictor:
    X, y = synthetic_data
    return SeverityPredictor().fit(X, y)


# ---------------------------------------------------------------------------
# test_fit_stores_cv_scores
# ---------------------------------------------------------------------------


def test_fit_stores_cv_scores(fitted_predictor):
    """fit() populates cv_scores_ with one accuracy value per fold."""
    assert fitted_predictor.cv_scores_ is not None
    assert len(fitted_predictor.cv_scores_) == 5
    assert all(0.0 <= s <= 1.0 for s in fitted_predictor.cv_scores_)


def test_fit_cv_scores_are_reasonable(fitted_predictor):
    """Mean CV accuracy should be above random chance (>0.5) on 200 samples."""
    assert fitted_predictor.cv_scores_.mean() > 0.5


# ---------------------------------------------------------------------------
# test_predict_returns_binary_array
# ---------------------------------------------------------------------------


def test_predict_returns_binary_array(fitted_predictor, synthetic_data):
    """predict() returns a 1-D int array containing only 0s and 1s."""
    X, _ = synthetic_data
    preds = fitted_predictor.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_predict_ignores_non_feature_columns(fitted_predictor, synthetic_data):
    """predict() silently drops cve_id and is_exploited if present."""
    X, y = synthetic_data
    X_with_extras = X.copy()
    X_with_extras["cve_id"] = "CVE-2024-00001"
    X_with_extras["is_exploited"] = y

    preds_clean = fitted_predictor.predict(X)
    preds_extras = fitted_predictor.predict(X_with_extras)
    np.testing.assert_array_equal(preds_clean, preds_extras)


# ---------------------------------------------------------------------------
# test_predict_proba_returns_probabilities_summing_to_one
# ---------------------------------------------------------------------------


def test_predict_proba_returns_probabilities_summing_to_one(fitted_predictor, synthetic_data):
    """predict_proba() returns (n_samples, 2) array; each row sums to 1."""
    X, _ = synthetic_data
    proba = fitted_predictor.predict_proba(X)

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)
    assert (proba >= 0).all() and (proba <= 1).all()


# ---------------------------------------------------------------------------
# test_explain_returns_shap_values_and_top_features
# ---------------------------------------------------------------------------


def test_explain_returns_shap_values_and_top_features(fitted_predictor, synthetic_data):
    """explain() returns correct keys with valid shapes and feature names."""
    X, _ = synthetic_data
    X_sample = X.iloc[:20]
    result = fitted_predictor.explain(X_sample)

    assert set(result.keys()) == {
        "shap_values",
        "feature_names",
        "top_positive_features",
        "top_negative_features",
    }

    shap_vals = result["shap_values"]
    assert isinstance(shap_vals, np.ndarray)
    assert shap_vals.shape == (20, _N_FEATURES)

    feature_names = result["feature_names"]
    assert feature_names == _FEATURE_COLS

    top_pos = result["top_positive_features"]
    top_neg = result["top_negative_features"]
    assert len(top_pos) == 3
    assert len(top_neg) == 3
    assert all(f in _FEATURE_COLS for f in top_pos)
    assert all(f in _FEATURE_COLS for f in top_neg)


def test_explain_top_features_are_distinct(fitted_predictor, synthetic_data):
    """Top positive and top negative feature lists have no duplicates."""
    X, _ = synthetic_data
    result = fitted_predictor.explain(X.iloc[:20])
    assert len(set(result["top_positive_features"])) == 3
    assert len(set(result["top_negative_features"])) == 3


# ---------------------------------------------------------------------------
# test_fit_raises_on_small_dataset
# ---------------------------------------------------------------------------


def test_fit_raises_on_small_dataset():
    """fit() raises ValueError when fewer than 50 training samples are provided."""
    X_small = pd.DataFrame(
        np.random.default_rng(0).random((30, _N_FEATURES)),
        columns=_FEATURE_COLS,
    )
    y_small = pd.Series(np.random.default_rng(0).integers(0, 2, 30))

    with pytest.raises(ValueError, match="50 samples"):
        SeverityPredictor().fit(X_small, y_small)


@pytest.mark.parametrize("n_samples", [0, 1, 49])
def test_fit_raises_for_various_small_sizes(n_samples):
    """fit() raises ValueError for any dataset smaller than 50 rows."""
    X = pd.DataFrame(
        np.zeros((n_samples, _N_FEATURES)),
        columns=_FEATURE_COLS,
    )
    y = pd.Series(np.zeros(n_samples, dtype=int))

    with pytest.raises(ValueError):
        SeverityPredictor().fit(X, y)


# ---------------------------------------------------------------------------
# test_model_registry_save_and_load_roundtrip
# ---------------------------------------------------------------------------


def test_model_registry_save_and_load_roundtrip(fitted_predictor, synthetic_data, tmp_path):
    """Saving and loading a model via ModelRegistry preserves predictions."""
    registry = ModelRegistry(models_dir=tmp_path)
    registry.save_model(fitted_predictor, "severity_predictor", "1.0")

    loaded: SeverityPredictor = registry.load_model("severity_predictor", "1.0")  # type: ignore[assignment]

    X, _ = synthetic_data
    np.testing.assert_array_equal(
        fitted_predictor.predict(X),
        loaded.predict(X),
    )


def test_model_registry_list_versions(fitted_predictor, tmp_path):
    """list_versions() returns all saved version strings for a model name."""
    registry = ModelRegistry(models_dir=tmp_path)
    registry.save_model(fitted_predictor, "severity_predictor", "1.0")
    registry.save_model(fitted_predictor, "severity_predictor", "2.0")
    # Different model name — should not appear
    registry.save_model(fitted_predictor, "other_model", "1.0")

    versions = registry.list_versions("severity_predictor")
    assert versions == ["1.0", "2.0"]
    assert registry.list_versions("other_model") == ["1.0"]


def test_model_registry_load_missing_raises(tmp_path):
    """load_model() raises FileNotFoundError for a non-existent version."""
    registry = ModelRegistry(models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.load_model("severity_predictor", "99.0")


def test_model_registry_list_versions_empty_dir(tmp_path):
    """list_versions() returns [] when the models directory is empty."""
    registry = ModelRegistry(models_dir=tmp_path)
    assert registry.list_versions("severity_predictor") == []
