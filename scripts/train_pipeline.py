"""One-time setup script: fetch CVEs, train ML model, build FAISS index."""

import random
import sys
import time
from pathlib import Path

# Ensure src/ is importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# Step 1 — Fetch CVEs
# ---------------------------------------------------------------------------

_SYNTHETIC_COUNT = 150


def _make_synthetic_cves(n: int = _SYNTHETIC_COUNT) -> list[dict]:
    """Generate synthetic CVE dicts when no real data is available."""
    random.seed(42)
    attack_vectors = ["NETWORK", "NETWORK", "NETWORK", "LOCAL", "ADJACENT_NETWORK", "PHYSICAL"]
    complexities = ["LOW", "LOW", "HIGH"]
    privs = ["NONE", "NONE", "LOW", "HIGH"]
    ui_choices = ["NONE", "NONE", "REQUIRED"]
    scopes = ["UNCHANGED", "UNCHANGED", "CHANGED"]
    impacts = ["NONE", "LOW", "HIGH", "HIGH"]
    cwes = ["CWE-79", "CWE-89", "CWE-787", "CWE-416", "CWE-78", "CWE-20", "CWE-22", "CWE-200"]

    cves = []
    for i in range(n):
        score = round(random.uniform(2.0, 10.0), 1)
        has_exploit = random.random() < 0.30
        refs = ["https://vendor.example.com/advisory"]
        if has_exploit:
            refs.append(f"https://github.com/poc-user/CVE-2024-{i + 10000}-poc")

        cves.append({
            "cve_id": f"CVE-2024-{i + 10000}",
            "description": (
                f"A vulnerability in component-{i} allows an attacker "
                f"to perform unauthorized actions on affected systems."
            ),
            "cvss_v3_score": score,
            "cvss_v3_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "attack_vector": random.choice(attack_vectors),
            "attack_complexity": random.choice(complexities),
            "privileges_required": random.choice(privs),
            "user_interaction": random.choice(ui_choices),
            "scope": random.choice(scopes),
            "confidentiality_impact": random.choice(impacts),
            "integrity_impact": random.choice(impacts),
            "availability_impact": random.choice(impacts),
            "cwe_ids": [random.choice(cwes)],
            "references": refs,
            "has_exploit_ref": has_exploit,
            "published_date": (
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T00:00:00.000"
            ),
            "last_modified_date": (
                f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}T00:00:00.000"
            ),
            "affected_products": [
                f"cpe:2.3:a:vendor{i}:product{i}:1.0:*:*:*:*:*:*:*"
            ],
        })

    return cves


def step1_fetch_cves() -> list[dict]:
    print("\nStep 1/4: Fetching CVEs from NVD...")
    t0 = time.perf_counter()

    from data.nvd_client import NVDClient

    client = NVDClient()

    # NVDClient loads from cache automatically if data/raw/cves_cache.json exists.
    # On cache hit it logs the source; we reflect that in our output.
    from data.nvd_client import _DEFAULT_CACHE_PATH

    if _DEFAULT_CACHE_PATH.exists():
        print(f"  Cache found — loading from {_DEFAULT_CACHE_PATH}")

    try:
        cves = client.fetch_cves(max_results=500)
    except Exception as exc:
        print(f"  NVD fetch error: {exc}")
        cves = []

    if cves:
        source = "cache" if _DEFAULT_CACHE_PATH.exists() else "NVD API"
        print(f"  Loaded {len(cves)} CVEs from {source}  ({_fmt(time.perf_counter() - t0)})")
        return cves

    # Fallback: generate synthetic training data
    print("  NVD returned 0 results — generating synthetic CVEs for training")
    cves = _make_synthetic_cves(_SYNTHETIC_COUNT)
    print(f"  Generated {len(cves)} synthetic CVEs  ({_fmt(time.perf_counter() - t0)})")
    print("  [Note: train on real NVD data for production use]")
    return cves


# ---------------------------------------------------------------------------
# Step 2 — Feature extraction
# ---------------------------------------------------------------------------


def step2_extract_features(cves: list[dict]):
    print("\nStep 2/4: Extracting features...")
    t0 = time.perf_counter()

    from data.feature_store import FeatureStore
    from data.preprocessor import CVEPreprocessor

    df = CVEPreprocessor().transform(cves)
    FeatureStore().save(df, "features")

    exploited = int(df["is_exploited"].sum())
    print(f"  {len(df)} rows × {len(df.columns)} features")
    print(f"  Exploited: {exploited} ({exploited / len(df) * 100:.1f}%)  "
          f"Not exploited: {len(df) - exploited}")
    print(f"  Saved to data/processed/features.csv  ({_fmt(time.perf_counter() - t0)})")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Train ML model
# ---------------------------------------------------------------------------


def step3_train_model(df):
    print("\nStep 3/4: Training ML model...")
    t0 = time.perf_counter()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from evaluation.metrics import evaluate_model
    from models.model_registry import ModelRegistry
    from models.severity_predictor import SeverityPredictor

    X: pd.DataFrame = df.drop(columns=["is_exploited", "cve_id"], errors="ignore")
    y: pd.Series = df["is_exploited"]

    if len(X) < 50:
        raise RuntimeError(
            f"Only {len(X)} samples — need at least 50 to train. "
            "Try increasing days_back in fetch_cves()."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    predictor = SeverityPredictor()
    predictor.fit(X_train, y_train)

    cv_mean = float(np.mean(predictor.cv_scores_))
    cv_std = float(np.std(predictor.cv_scores_))
    print(f"  CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}  "
          f"(folds: {[f'{s:.3f}' for s in predictor.cv_scores_]})")

    metrics = evaluate_model(predictor, X_test, y_test)
    print(f"  Test accuracy : {metrics['accuracy']:.3f}")
    print(f"  Test precision: {metrics['precision']:.3f}")
    print(f"  Test recall   : {metrics['recall']:.3f}")
    print(f"  Test F1       : {metrics['f1_score']:.3f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC       : {metrics['roc_auc']:.3f}")
    print("\n  Classification report:")
    for line in metrics["classification_report"].splitlines():
        print(f"    {line}")

    registry = ModelRegistry()
    registry.save_model(predictor, "severity_predictor", "v1")
    print(f"\n  Model saved as severity_predictor_vv1.joblib  ({_fmt(time.perf_counter() - t0)})")
    return predictor


# ---------------------------------------------------------------------------
# Step 4 — Build FAISS index
# ---------------------------------------------------------------------------


def step4_build_index() -> None:
    print("\nStep 4/4: Building knowledge base index...")
    t0 = time.perf_counter()

    from rag.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    kb.build_index()
    stats = kb.get_stats()
    print(f"  Documents : {stats['doc_count']}")
    print(f"  Chunks    : {stats['chunk_count']}")
    print(f"  Vectors   : {stats['index_size']}")
    print(f"  Index saved to data/processed/faiss_index/  ({_fmt(time.perf_counter() - t0)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    overall_start = time.perf_counter()
    print("=" * 60)
    print("  ThreatLens AI — Training Pipeline")
    print("=" * 60)

    try:
        cves = step1_fetch_cves()
    except RuntimeError as exc:
        print(f"\n[ERROR] Step 1 failed: {exc}")
        sys.exit(1)

    try:
        df = step2_extract_features(cves)
    except Exception as exc:
        print(f"\n[ERROR] Step 2 failed: {exc}")
        sys.exit(1)

    try:
        step3_train_model(df)
    except Exception as exc:
        print(f"\n[ERROR] Step 3 failed: {exc}")
        sys.exit(1)

    try:
        step4_build_index()
    except Exception as exc:
        print(f"\n[ERROR] Step 4 failed: {exc}")
        sys.exit(1)

    total = _fmt(time.perf_counter() - overall_start)
    print("\n" + "=" * 60)
    print(f"  Training complete!  (total: {total})")
    print("=" * 60)
    print("\nStart the server with:")
    print("  uvicorn src.api.main:app --reload")
    print()


if __name__ == "__main__":
    main()
