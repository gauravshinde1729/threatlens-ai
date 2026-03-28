"""POST /analyze — full pipeline: NVD fetch → ML predict → RAG playbook."""

import logging
import time

from fastapi import APIRouter, HTTPException

from api.dependencies import app_state
from api.routes.playbook import _parse_sections
from api.routes.predict import _fetch_single_cve, _risk_level
from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    PlaybookResponse,
    SeverityResponse,
    ShapExplanation,
)
from data.preprocessor import _REQUIRED_COLUMNS, CVEPreprocessor
from rag.playbook_generator import PlaybookGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

_FEATURE_COLS = [c for c in _REQUIRED_COLUMNS if c not in ("cve_id", "is_exploited")]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Full ThreatLens pipeline for a single CVE.

    Steps:
    1. Fetch CVE from NVD API
    2. Extract features with CVEPreprocessor
    3. Predict exploit probability with SeverityPredictor
    4. Generate SHAP explanation
    5. Retrieve relevant KB docs and generate remediation playbook

    Returns combined severity assessment and remediation playbook.
    """
    if not app_state.model_loaded or app_state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run training pipeline first.",
        )
    if not app_state.index_loaded or app_state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run training pipeline first.",
        )

    t0 = time.perf_counter()
    cve_id = request.cve_id

    # Step 1 — Fetch CVE
    cve_data = _fetch_single_cve(cve_id)

    # Step 2 — Feature extraction
    preprocessor = CVEPreprocessor()
    df_full = preprocessor.transform([cve_data])
    X = df_full[[c for c in _FEATURE_COLS if c in df_full.columns]]

    # Step 3 — ML prediction
    proba = app_state.predictor.predict_proba(X)
    exploit_prob = float(proba[0][1])

    # Step 4 — SHAP explanation
    explanation = app_state.predictor.explain(X)

    severity = SeverityResponse(
        cve_id=cve_id,
        cvss_score=cve_data.get("cvss_v3_score"),
        exploit_probability=exploit_prob,
        risk_level=_risk_level(exploit_prob),
        shap_explanation=ShapExplanation(
            top_positive_features=explanation["top_3_positive"],
            top_negative_features=explanation["top_3_negative"],
        ),
    )

    # Step 5 — RAG playbook
    docs = app_state.retriever.retrieve_for_cve(cve_data, top_k=5)
    ml_prediction = {
        "exploit_probability": exploit_prob,
        "predicted_label": int(exploit_prob >= 0.5),
        "confidence": _risk_level(exploit_prob),
    }

    try:
        generator = PlaybookGenerator(app_state.retriever)
        result = generator.generate(cve_data, ml_prediction, docs)
        sections = _parse_sections(result["playbook"])
        sources = result["sources"]
    except OSError as exc:
        # GROQ_API_KEY not set — return empty playbook rather than crashing
        logger.warning("Playbook generation skipped: %s", exc)
        sections = {"error": "GROQ_API_KEY not configured — playbook unavailable"}
        sources = []

    playbook = PlaybookResponse(cve_id=cve_id, playbook=sections, sources=sources)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("analyze %s complete in %.1fms risk=%s", cve_id, latency_ms, severity.risk_level)

    return AnalyzeResponse(cve_id=cve_id, severity=severity, playbook=playbook)
