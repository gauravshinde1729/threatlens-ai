"""Population Stability Index (PSI) based feature drift detector."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PSI_MODERATE = 0.1
_PSI_SIGNIFICANT = 0.25
_N_BINS = 10
_EPS = 1e-8  # avoid log(0)


def _psi(reference: np.ndarray, current: np.ndarray, n_bins: int = _N_BINS) -> float:
    """Compute PSI between a reference and current distribution.

    Bins are determined from the reference distribution.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Use quantile-based bins from reference to avoid empty bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref, quantiles))

    # Need at least 2 unique edges to form bins
    if len(bin_edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / (len(ref) + _EPS)
    cur_pct = cur_counts / (len(cur) + _EPS)

    # Replace zeros to avoid log(0)
    ref_pct = np.where(ref_pct == 0, _EPS, ref_pct)
    cur_pct = np.where(cur_pct == 0, _EPS, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_status(psi_value: float) -> str:
    if psi_value < _PSI_MODERATE:
        return "no_drift"
    if psi_value < _PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


class DriftDetector:
    """Detect feature distribution drift using Population Stability Index."""

    def __init__(self) -> None:
        self._reference: pd.DataFrame | None = None

    def set_reference(self, X: pd.DataFrame) -> None:
        """Store the reference (training) distribution.

        Args:
            X: DataFrame of numeric features from the training set.
        """
        self._reference = X.select_dtypes(include="number").copy()
        logger.info(
            "Reference distribution set: %d samples, %d features",
            len(self._reference),
            self._reference.shape[1],
        )

    def detect_drift(self, X_current: pd.DataFrame) -> dict:
        """Compute PSI for each feature and return drift status.

        Args:
            X_current: DataFrame of numeric features from the current window.

        Returns:
            Dict with per-feature PSI results and overall_drift flag:
            {
                "feature_a": {"psi": 0.05, "status": "no_drift"},
                ...
                "overall_drift": False
            }
        """
        if self._reference is None:
            raise RuntimeError("Call set_reference() before detect_drift()")

        X_num = X_current.select_dtypes(include="number")
        common_cols = [c for c in self._reference.columns if c in X_num.columns]

        if not common_cols:
            raise ValueError("No common numeric columns between reference and current data")

        results: dict = {}
        any_significant = False

        for col in common_cols:
            ref_vals = self._reference[col].values
            cur_vals = X_num[col].values
            psi_val = _psi(ref_vals, cur_vals)
            status = _psi_status(psi_val)
            results[col] = {"psi": round(psi_val, 6), "status": status}
            if status == "significant":
                any_significant = True

        results["overall_drift"] = any_significant
        logger.info(
            "Drift detection: %d features checked, overall_drift=%s",
            len(common_cols),
            any_significant,
        )
        return results
