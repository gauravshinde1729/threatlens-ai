"""Predicts whether a CVE will be exploited in the wild."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_config.yaml"
_MIN_TRAIN_SAMPLES = 50

# Columns that are not model input features
_NON_FEATURE_COLS = {"cve_id", "is_exploited"}


def _load_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        with _CONFIG_PATH.open() as f:
            return yaml.safe_load(f) or {}
    logger.warning("model_config.yaml not found — using defaults")
    return {}


class SeverityPredictor:
    """Ensemble classifier (RandomForest + XGBoost) for CVE exploit prediction."""

    def __init__(self) -> None:
        cfg = _load_config()
        rf_cfg = cfg.get("random_forest", {})
        xgb_cfg = cfg.get("xgboost", {})
        cv_cfg = cfg.get("cross_validation", {})

        rf = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 100),
            max_depth=rf_cfg.get("max_depth", None),
            min_samples_split=rf_cfg.get("min_samples_split", 2),
            random_state=rf_cfg.get("random_state", 42),
        )
        xgb = XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 100),
            max_depth=xgb_cfg.get("max_depth", 6),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            eval_metric=xgb_cfg.get("eval_metric", "logloss"),
            random_state=xgb_cfg.get("random_state", 42),
            verbosity=0,
        )

        self._model = VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
        )
        self._n_splits: int = cv_cfg.get("n_splits", 5)
        self._cv_shuffle: bool = cv_cfg.get("shuffle", True)
        self.cv_scores_: np.ndarray | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeverityPredictor":
        """Train the ensemble and run stratified cross-validation.

        Args:
            X: Feature DataFrame (must not contain cve_id or is_exploited).
            y: Binary target Series (1 = exploited, 0 = not exploited).

        Returns:
            self
        """
        X = self._drop_non_features(X)

        if len(X) < _MIN_TRAIN_SAMPLES:
            raise ValueError(
                f"Training requires at least {_MIN_TRAIN_SAMPLES} samples, "
                f"got {len(X)}. Collect more CVE data before fitting."
            )

        self._feature_names = list(X.columns)
        logger.info("Fitting SeverityPredictor on %d samples, %d features", len(X), len(self._feature_names))

        cv = StratifiedKFold(n_splits=self._n_splits, shuffle=self._cv_shuffle, random_state=42)
        self.cv_scores_ = cross_val_score(self._model, X, y, cv=cv, scoring="accuracy")
        logger.info(
            "CV accuracy: %.3f ± %.3f",
            self.cv_scores_.mean(),
            self.cv_scores_.std(),
        )

        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 or 1).

        Args:
            X: Feature DataFrame.

        Returns:
            1-D array of predicted labels.
        """
        return self._model.predict(self._drop_non_features(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return exploitation probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape (n_samples, 2) — [prob_not_exploited, prob_exploited].
        """
        return self._model.predict_proba(self._drop_non_features(X))

    def explain(self, X: pd.DataFrame) -> dict[str, Any]:
        """Generate SHAP explanations using the RandomForest sub-model.

        Args:
            X: Feature DataFrame (same columns used during fit).

        Returns:
            Dict with keys:
                shap_values, feature_names,
                top_positive_features, top_negative_features
        """
        X = self._drop_non_features(X)
        rf_estimator = self._model.estimators_[0]  # RF is first in estimators list
        explainer = shap.TreeExplainer(rf_estimator)
        raw = explainer.shap_values(X)

        # Normalise shape: older SHAP returns list [cls0, cls1]; newer returns 3-D array
        if isinstance(raw, list):
            values_cls1 = np.array(raw[1])          # (n_samples, n_features)
        else:
            values_cls1 = raw[:, :, 1]              # (n_samples, n_features, n_classes)

        feature_names = list(X.columns)
        mean_shap = values_cls1.mean(axis=0)        # (n_features,)

        # Top 3 features pushing toward exploitation (highest positive mean SHAP)
        pos_idx = np.argsort(mean_shap)[::-1][:3]
        top_positive = [feature_names[i] for i in pos_idx]

        # Top 3 features pushing against exploitation (most negative mean SHAP)
        neg_idx = np.argsort(mean_shap)[:3]
        top_negative = [feature_names[i] for i in neg_idx]

        return {
            "shap_values": values_cls1,
            "feature_names": feature_names,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_non_features(X: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in _NON_FEATURE_COLS if c in X.columns]
        return X.drop(columns=cols_to_drop) if cols_to_drop else X
