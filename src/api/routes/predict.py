"""POST /predict — ML severity prediction without LLM playbook generation."""

import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import app_state
from api.schemas import PredictRequest, SeverityResponse, ShapExplanation
from data.nvd_client import NVDClient
from data.preprocessor import _REQUIRED_COLUMNS, CVEPreprocessor

router = APIRouter()
logger = logging.getLogger(__name__)

_FEATURE_COLS = [c for c in _REQUIRED_COLUMNS if c not in ("cve_id", "is_exploited")]


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


@router.post("/predict", response_model=SeverityResponse)
async def predict(request: PredictRequest) -> SeverityResponse:
    """Predict exploit probability for a CVE using the trained ML ensemble.

    Accepts either a CVE ID (fetched from NVD) or a pre-computed feature dict.
    Returns binary risk level and SHAP-based feature attribution.
    """
    if not app_state.model_loaded or app_state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run training pipeline first.",
        )

    t0 = time.perf_counter()

    # Build feature row
    if request.features:
        row = {col: request.features.get(col, 0) for col in _FEATURE_COLS}
        cvss_score = request.features.get("cvss_v3_score")
    else:
        cve_data = _fetch_single_cve(request.cve_id)
        preprocessor = CVEPreprocessor()
        df_full = preprocessor.transform([cve_data])
        cvss_score = cve_data.get("cvss_v3_score")
        row = {col: df_full.iloc[0][col] for col in _FEATURE_COLS if col in df_full.columns}

    X = pd.DataFrame([row])[_FEATURE_COLS]

    proba = app_state.predictor.predict_proba(X)
    exploit_prob = float(proba[0][1])

    explanation = app_state.predictor.explain(X)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "predict %s: prob=%.3f risk=%s latency=%.1fms",
        request.cve_id,
        exploit_prob,
        _risk_level(exploit_prob),
        latency_ms,
    )

    return SeverityResponse(
        cve_id=request.cve_id,
        cvss_score=float(cvss_score) if cvss_score is not None else None,
        exploit_probability=exploit_prob,
        risk_level=_risk_level(exploit_prob),
        shap_explanation=ShapExplanation(
            top_positive_features=explanation["top_3_positive"],
            top_negative_features=explanation["top_3_negative"],
        ),
    )


def _fetch_single_cve(cve_id: str) -> dict:
    """Fetch and return the first matching CVE dict from NVD."""
    client = NVDClient()
    # Use a narrow 1-day window trick: fetch by lastModified recent range;
    # NVD 2.0 doesn't support filter by cve_id directly in basic requests,
    # so we fetch recent CVEs and filter, or fall back to a wide window.
    cves = client.fetch_cves(days_back=3650, max_results=500)
    match = next((c for c in cves if c.get("cve_id") == cve_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"{cve_id} not found in NVD (last 10 years)")
    return match
