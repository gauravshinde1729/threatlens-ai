"""Parquet-backed feature store for processed CVE feature DataFrames."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


class FeatureStore:
    """Save and load feature DataFrames from a Parquet store directory."""

    def __init__(self, store_dir: str | Path = _DEFAULT_STORE_DIR) -> None:
        self.store_dir = Path(store_dir)

    def save(self, df: pd.DataFrame, name: str) -> Path:
        """Persist *df* as ``<store_dir>/<name>.csv``.

        Args:
            df: Feature DataFrame to persist.
            name: Logical dataset name (no extension needed).

        Returns:
            Path to the written file.
        """
        self.store_dir.mkdir(parents=True, exist_ok=True)
        path = self.store_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("FeatureStore: saved '%s' (%d rows) → %s", name, len(df), path)
        return path

    def load(self, name: str) -> pd.DataFrame:
        """Load a feature DataFrame from ``<store_dir>/<name>.csv``.

        Args:
            name: Logical dataset name (no extension needed).

        Returns:
            Loaded DataFrame.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        path = self.store_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Feature store entry not found: {path}")
        df = pd.read_csv(path)
        logger.info("FeatureStore: loaded '%s' (%d rows) ← %s", name, len(df), path)
        return df

    def exists(self, name: str) -> bool:
        """Return True if ``<name>.csv`` exists in the store."""
        return (self.store_dir / f"{name}.csv").exists()
