"""Prediction performance tracker backed by a local JSON log file."""

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "metrics" / "predictions.json"
)

_RISK_LEVELS = ("HIGH", "MEDIUM", "LOW")


class PerformanceTracker:
    """Thread-safe tracker that appends prediction records to a JSON file."""

    def __init__(self, metrics_path: str | Path = _DEFAULT_METRICS_PATH) -> None:
        self.metrics_path = Path(metrics_path)
        self._lock = threading.Lock()
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.metrics_path.exists():
            self._write_records([])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        cve_id: str,
        prediction: dict[str, Any],
        latency_ms: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a prediction record to the metrics log.

        Args:
            cve_id: CVE identifier.
            prediction: Dict containing at least exploit_probability and risk_level.
            latency_ms: Request latency in milliseconds.
            timestamp: Event timestamp (defaults to now UTC).
        """
        ts = (timestamp or datetime.now(tz=UTC)).isoformat()
        record = {
            "cve_id": cve_id,
            "exploit_probability": prediction.get("exploit_probability"),
            "risk_level": prediction.get("risk_level", "UNKNOWN"),
            "latency_ms": latency_ms,
            "timestamp": ts,
            "error": prediction.get("error"),
        }
        with self._lock:
            records = self._read_records()
            records.append(record)
            self._write_records(records)
        logger.debug("Logged prediction: cve_id=%s risk=%s", cve_id, record["risk_level"])

    def get_metrics(self, last_n_hours: int = 24) -> dict[str, Any]:
        """Aggregate metrics over the last N hours.

        Args:
            last_n_hours: Time window in hours (default 24).

        Returns:
            Dict with total_predictions, avg_latency_ms, error_rate,
            and prediction_distribution (counts per risk level).
        """
        cutoff = datetime.now(tz=UTC) - timedelta(hours=last_n_hours)
        with self._lock:
            records = self._read_records()

        window = [r for r in records if _parse_ts(r.get("timestamp")) >= cutoff]

        total = len(window)
        if total == 0:
            return {
                "total_predictions": 0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
                "prediction_distribution": {lvl: 0 for lvl in _RISK_LEVELS},
            }

        errors = sum(1 for r in window if r.get("error") is not None)
        latencies = [r["latency_ms"] for r in window if r.get("latency_ms") is not None]
        distribution = {lvl: 0 for lvl in _RISK_LEVELS}
        for r in window:
            lvl = r.get("risk_level", "LOW")
            if lvl in distribution:
                distribution[lvl] += 1

        return {
            "total_predictions": total,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "error_rate": round(errors / total, 4),
            "prediction_distribution": distribution,
        }

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _read_records(self) -> list[dict]:
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write_records(self, records: list[dict]) -> None:
        self.metrics_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _parse_ts(ts_str: str | None) -> datetime:
    if ts_str is None:
        return datetime.min.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=UTC)
