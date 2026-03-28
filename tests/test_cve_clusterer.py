"""Tests for src/models/cve_clusterer.py."""

import numpy as np
import pandas as pd
import pytest

from models.cve_clusterer import CVEClusterer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clusterable_df(n_per_cluster: int = 30, n_clusters: int = 3, seed: int = 0) -> pd.DataFrame:
    """Build a DataFrame with clear cluster structure."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_clusters):
        center = np.array([i * 10.0, i * 10.0, i * 5.0])
        data = rng.normal(loc=center, scale=0.3, size=(n_per_cluster, 3))
        frames.append(data)
    X = np.vstack(frames)
    return pd.DataFrame(X, columns=["cvss_v3_score", "description_length", "reference_count"])


def _make_sparse_df(n: int = 50, seed: int = 1) -> pd.DataFrame:
    """Build a DataFrame where all points are too far apart to cluster."""
    rng = np.random.default_rng(seed)
    # Spread points very far apart so DBSCAN marks everything as noise
    data = rng.uniform(low=0, high=1000, size=(n, 3))
    return pd.DataFrame(data, columns=["cvss_v3_score", "description_length", "reference_count"])


@pytest.fixture()
def clusterable_df() -> pd.DataFrame:
    return _make_clusterable_df()


# ---------------------------------------------------------------------------
# test_fit_predict_returns_labels
# ---------------------------------------------------------------------------


def test_fit_predict_returns_labels(clusterable_df):
    """fit_predict() returns an integer array of shape (n_samples,)."""
    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(clusterable_df)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(clusterable_df),)
    assert labels.dtype.kind == "i"  # integer type


def test_fit_predict_finds_expected_clusters(clusterable_df):
    """fit_predict() identifies the 3 well-separated clusters."""
    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(clusterable_df)

    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 3


def test_fit_predict_drops_non_numeric_columns():
    """fit_predict() ignores string/ID columns without raising."""
    df = _make_clusterable_df()
    df["cve_id"] = [f"CVE-2024-{i:05d}" for i in range(len(df))]
    df["is_exploited"] = np.random.default_rng(0).integers(0, 2, len(df))

    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(df)

    assert labels.shape == (len(df),)


# ---------------------------------------------------------------------------
# test_cluster_summary_structure
# ---------------------------------------------------------------------------


def test_cluster_summary_structure(clusterable_df):
    """get_cluster_summary() returns a list with one dict per non-noise cluster."""
    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(clusterable_df)
    summaries = clusterer.get_cluster_summary(clusterable_df, labels)

    n_clusters = len(set(labels) - {-1})
    assert len(summaries) == n_clusters

    for summary in summaries:
        assert "cluster_id" in summary
        assert "size" in summary
        assert "top_3_features" in summary
        assert "representative_cve_indices" in summary

        assert isinstance(summary["cluster_id"], int)
        assert summary["cluster_id"] != -1
        assert summary["size"] > 0
        assert len(summary["top_3_features"]) == 3
        assert len(summary["representative_cve_indices"]) <= 3

        # Representative indices are valid row positions
        for idx in summary["representative_cve_indices"]:
            assert 0 <= idx < len(clusterable_df)

        # top_3_features reference actual columns
        num_cols = clusterable_df.select_dtypes(include="number").columns.tolist()
        for feat in summary["top_3_features"]:
            assert feat in num_cols


def test_cluster_summary_sizes_add_up(clusterable_df):
    """Sum of cluster sizes equals total non-noise points."""
    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(clusterable_df)
    summaries = clusterer.get_cluster_summary(clusterable_df, labels)

    total_from_summaries = sum(s["size"] for s in summaries)
    total_non_noise = int((labels != -1).sum())
    assert total_from_summaries == total_non_noise


def test_cluster_summary_representative_indices_are_in_cluster(clusterable_df):
    """Each representative index belongs to the correct cluster."""
    clusterer = CVEClusterer(eps=1.0, min_samples=3)
    labels = clusterer.fit_predict(clusterable_df)
    summaries = clusterer.get_cluster_summary(clusterable_df, labels)

    for summary in summaries:
        cid = summary["cluster_id"]
        for idx in summary["representative_cve_indices"]:
            assert labels[idx] == cid


# ---------------------------------------------------------------------------
# test_handles_all_noise_gracefully
# ---------------------------------------------------------------------------


def test_handles_all_noise_gracefully():
    """get_cluster_summary() returns [] when DBSCAN assigns all points as noise."""
    sparse_df = _make_sparse_df()
    # Very tight eps so nothing clusters
    clusterer = CVEClusterer(eps=0.001, min_samples=100)
    labels = clusterer.fit_predict(sparse_df)

    assert (labels == -1).all(), "Expected all noise with these hyperparameters"
    summaries = clusterer.get_cluster_summary(sparse_df, labels)
    assert summaries == []


def test_fit_predict_all_noise_returns_minus_one_array():
    """fit_predict() returns all -1 labels when no cluster can form."""
    sparse_df = _make_sparse_df()
    clusterer = CVEClusterer(eps=0.001, min_samples=100)
    labels = clusterer.fit_predict(sparse_df)

    assert set(labels).issubset({-1})
