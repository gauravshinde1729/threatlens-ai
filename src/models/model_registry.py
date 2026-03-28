"""Joblib-backed model registry for versioned model persistence."""

import logging
import re
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


class ModelRegistry:
    """Save, load, and list versioned ML models stored as joblib files."""

    def __init__(self, models_dir: str | Path = _DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)

    def save_model(self, model: object, name: str, version: str) -> Path:
        """Persist *model* to ``<models_dir>/<name>_v<version>.joblib``.

        Args:
            model: Any picklable Python object (sklearn estimator, pipeline, etc.).
            name: Logical model name (e.g. "severity_predictor").
            version: Version string (e.g. "1.0", "20240315").

        Returns:
            Path to the written file.
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)
        path = self.models_dir / f"{name}_v{version}.joblib"
        joblib.dump(model, path)
        logger.info("ModelRegistry: saved '%s' v%s → %s", name, version, path)
        return path

    def load_model(self, name: str, version: str) -> object:
        """Load a model from ``<models_dir>/<name>_v<version>.joblib``.

        Args:
            name: Logical model name.
            version: Version string.

        Returns:
            Deserialised model object.

        Raises:
            FileNotFoundError: If the joblib file does not exist.
        """
        path = self.models_dir / f"{name}_v{version}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found in registry: {path}. "
                f"Available versions: {self.list_versions(name)}"
            )
        model = joblib.load(path)
        logger.info("ModelRegistry: loaded '%s' v%s ← %s", name, version, path)
        return model

    def list_versions(self, name: str) -> list[str]:
        """Return all available version strings for a given model name.

        Args:
            name: Logical model name.

        Returns:
            Sorted list of version strings (e.g. ["1.0", "2.0"]).
        """
        if not self.models_dir.exists():
            return []
        pattern = re.compile(rf"^{re.escape(name)}_v(.+)\.joblib$")
        versions = []
        for path in self.models_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                versions.append(match.group(1))
        return sorted(versions)
