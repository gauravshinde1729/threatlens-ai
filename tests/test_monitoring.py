"""Tests for drift_detector.py and performance_tracker.py."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from monitoring.drift_detector import DriftDetector
from monitoring.performance_tracker import PerformanceTracker

# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector_with_reference() -> DriftDetector:
    rng = np.random.default_rng(42)
    X_ref = pd.DataFrame(
        {
            "cvss_v3_score": rng.uniform(0, 10, 500),
            "description_length": rng.integers(50, 500, 500).astype(float),
            "reference_count": rng.integers(0, 20, 500).astype(float),
        }
    )
    detector = DriftDetector()
    detector.set_reference(X_ref)
    return detector


def test_drift_detector_passes_on_stable_data(detector_with_reference):
    """Similar distribution to reference produces no_drift status."""
    rng = np.random.default_rng(99)
    X_stable = pd.DataFrame(
        {
            "cvss_v3_score": rng.uniform(0, 10, 300),
            "description_length": rng.integers(50, 500, 300).astype(float),
            "reference_count": rng.integers(0, 20, 300).astype(float),
        }
    )
    result = detector_with_reference.detect_drift(X_stable)

    assert "overall_drift" in result
    assert result["overall_drift"] is False
    for col in ("cvss_v3_score", "description_length", "reference_count"):
        assert col in result
        assert result[col]["status"] in ("no_drift", "moderate")


def test_drift_detector_flags_significant_drift(detector_with_reference):
    """Distribution shift to very different range triggers significant drift."""
    rng = np.random.default_rng(7)
    # Dramatically different range — reference is U[0,10], this is U[50,100]
    X_shifted = pd.DataFrame(
        {
            "cvss_v3_score": rng.uniform(50, 100, 300),
            "description_length": rng.integers(50, 500, 300).astype(float),
            "reference_count": rng.integers(0, 20, 300).astype(float),
        }
    )
    result = detector_with_reference.detect_drift(X_shifted)

    assert result["overall_drift"] is True
    assert result["cvss_v3_score"]["status"] == "significant"
    assert result["cvss_v3_score"]["psi"] >= 0.25


def test_drift_detector_result_structure(detector_with_reference):
    """detect_drift() returns dicts with psi (float) and status (str) per feature."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"cvss_v3_score": rng.uniform(0, 10, 100)})
    result = detector_with_reference.detect_drift(X)

    assert "cvss_v3_score" in result
    assert "overall_drift" in result
    entry = result["cvss_v3_score"]
    assert isinstance(entry["psi"], float)
    assert entry["psi"] >= 0.0
    assert entry["status"] in ("no_drift", "moderate", "significant")


def test_drift_detector_raises_before_reference_set():
    """detect_drift() raises RuntimeError when set_reference() was not called."""
    detector = DriftDetector()
    X = pd.DataFrame({"cvss_v3_score": [1.0, 2.0, 3.0]})
    with pytest.raises(RuntimeError, match="set_reference"):
        detector.detect_drift(X)


def test_drift_detector_ignores_non_numeric_columns(detector_with_reference):
    """Non-numeric columns are silently skipped."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame(
        {
            "cvss_v3_score": rng.uniform(0, 10, 100),
            "cve_id": [f"CVE-2024-{i:05d}" for i in range(100)],
        }
    )
    result = detector_with_reference.detect_drift(X)
    assert "cve_id" not in result


@pytest.mark.parametrize("psi,expected_status", [
    (0.05, "no_drift"),
    (0.15, "moderate"),
    (0.30, "significant"),
])
def test_psi_status_thresholds(psi, expected_status):
    """PSI thresholds map to correct status strings."""
    from monitoring.drift_detector import _psi_status
    assert _psi_status(psi) == expected_status


# ---------------------------------------------------------------------------
# PerformanceTracker
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker(tmp_path) -> PerformanceTracker:
    return PerformanceTracker(metrics_path=tmp_path / "predictions.json")


def test_performance_tracker_logs_and_retrieves(tracker):
    """log_prediction() persists a record retrievable via get_metrics()."""
    tracker.log_prediction(
        cve_id="CVE-2024-00001",
        prediction={"exploit_probability": 0.9, "risk_level": "HIGH"},
        latency_ms=42.5,
    )
    metrics = tracker.get_metrics(last_n_hours=24)

    assert metrics["total_predictions"] == 1
    assert metrics["avg_latency_ms"] == pytest.approx(42.5)
    assert metrics["prediction_distribution"]["HIGH"] == 1


def test_performance_tracker_multiple_predictions(tracker):
    """get_metrics() aggregates across multiple logged predictions."""
    for i in range(5):
        tracker.log_prediction(
            cve_id=f"CVE-2024-{i:05d}",
            prediction={"exploit_probability": 0.3, "risk_level": "MEDIUM"},
            latency_ms=float(10 + i),
        )
    metrics = tracker.get_metrics(last_n_hours=24)

    assert metrics["total_predictions"] == 5
    assert metrics["prediction_distribution"]["MEDIUM"] == 5
    assert metrics["avg_latency_ms"] == pytest.approx(12.0)  # mean of 10..14


def test_performance_tracker_time_window(tracker):
    """Predictions outside the time window are excluded from get_metrics()."""
    old_ts = datetime.now(tz=UTC) - timedelta(hours=48)
    tracker.log_prediction(
        cve_id="CVE-2024-OLD",
        prediction={"exploit_probability": 0.8, "risk_level": "HIGH"},
        latency_ms=10.0,
        timestamp=old_ts,
    )
    # Recent prediction
    tracker.log_prediction(
        cve_id="CVE-2024-NEW",
        prediction={"exploit_probability": 0.2, "risk_level": "LOW"},
        latency_ms=20.0,
    )

    metrics = tracker.get_metrics(last_n_hours=24)
    assert metrics["total_predictions"] == 1
    assert metrics["prediction_distribution"]["LOW"] == 1
    assert metrics["prediction_distribution"]["HIGH"] == 0


def test_performance_tracker_empty_window(tracker):
    """get_metrics() returns zero counts when no predictions in window."""
    metrics = tracker.get_metrics(last_n_hours=1)

    assert metrics["total_predictions"] == 0
    assert metrics["avg_latency_ms"] == 0.0
    assert metrics["error_rate"] == 0.0
    assert all(v == 0 for v in metrics["prediction_distribution"].values())


def test_performance_tracker_error_rate(tracker):
    """error_rate reflects proportion of predictions that have an error field."""
    tracker.log_prediction(
        "CVE-A", {"risk_level": "HIGH", "error": "timeout"}, latency_ms=5.0
    )
    tracker.log_prediction(
        "CVE-B", {"risk_level": "LOW"}, latency_ms=5.0
    )
    metrics = tracker.get_metrics(last_n_hours=24)
    assert metrics["error_rate"] == pytest.approx(0.5)


def test_performance_tracker_metrics_file_persists(tmp_path):
    """A second tracker instance reads records written by the first."""
    path = tmp_path / "shared.json"
    t1 = PerformanceTracker(metrics_path=path)
    t1.log_prediction("CVE-X", {"risk_level": "HIGH"}, latency_ms=1.0)

    t2 = PerformanceTracker(metrics_path=path)
    assert t2.get_metrics()["total_predictions"] == 1
