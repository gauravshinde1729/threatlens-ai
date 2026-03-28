"""Unsupervised CVE grouping via DBSCAN on scaled numeric features."""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that carry no numeric signal for clustering
_NON_NUMERIC_COLS = {"cve_id", "is_exploited"}


class CVEClusterer:
    """DBSCAN-based clusterer for identifying vulnerability campaigns."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self._eps = eps
        self._min_samples = min_samples
        self._scaler = StandardScaler()
        self._dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Scale numeric features then run DBSCAN.

        Args:
            X: Feature DataFrame (may include non-numeric columns — they
               are dropped before clustering).

        Returns:
            Integer cluster label array of shape (n_samples,).
            -1 indicates a noise point.
        """
        X_num = self._select_numeric(X)
        logger.info(
            "Clustering %d CVEs on %d numeric features (eps=%.2f, min_samples=%d)",
            len(X_num),
            X_num.shape[1],
            self._eps,
            self._min_samples,
        )
        X_scaled = self._scaler.fit_transform(X_num)
        labels: np.ndarray = self._dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        logger.info("Found %d clusters, %d noise points", n_clusters, n_noise)
        return labels

    def get_cluster_summary(
        self, X: pd.DataFrame, labels: np.ndarray
    ) -> list[dict]:
        """Summarise each cluster with statistics and representative members.

        Args:
            X: The same DataFrame passed to fit_predict.
            labels: Cluster labels returned by fit_predict.

        Returns:
            List of dicts (one per non-noise cluster), each containing:
                - cluster_id (int)
                - size (int)
                - top_3_features (list[str]): features with highest mean value
                - representative_cve_indices (list[int]): 3 row indices closest
                  to the cluster centroid
        """
        X_num = self._select_numeric(X)
        unique_clusters = sorted(c for c in set(labels) if c != -1)

        if not unique_clusters:
            logger.warning("No clusters found — all points are noise")
            return []

        summaries = []
        for cluster_id in unique_clusters:
            mask = labels == cluster_id
            cluster_df = X_num[mask]
            indices = np.where(mask)[0]

            # Top 3 features by mean value within the cluster
            means = cluster_df.mean()
            top_3_features = means.nlargest(3).index.tolist()

            # 3 rows closest to the centroid (L2 distance)
            centroid = means.values
            distances = np.linalg.norm(cluster_df.values - centroid, axis=1)
            closest_pos = np.argsort(distances)[: min(3, len(indices))]
            representative_indices = indices[closest_pos].tolist()

            summaries.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": int(mask.sum()),
                    "top_3_features": top_3_features,
                    "representative_cve_indices": representative_indices,
                }
            )

        return summaries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_numeric(X: pd.DataFrame) -> pd.DataFrame:
        """Return only numeric columns, excluding known non-feature cols."""
        X_clean = X.drop(columns=[c for c in _NON_NUMERIC_COLS if c in X.columns])
        return X_clean.select_dtypes(include="number")
