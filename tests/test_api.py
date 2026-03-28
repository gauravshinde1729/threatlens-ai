"""Tests for the FastAPI REST API endpoints."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers to build mock state
# ---------------------------------------------------------------------------


def _make_mock_predictor():
    """Minimal SeverityPredictor mock."""
    pred = MagicMock()
    pred.predict_proba.return_value = np.array([[0.08, 0.92]])
    pred.predict.return_value = np.array([1])
    pred.explain.return_value = {
        "shap_values": np.zeros((1, 30)),
        "feature_names": [f"f{i}" for i in range(30)],
        "top_3_positive": [("has_exploit_ref", 0.31), ("cvss_v3_score", 0.18), ("cwe_787", 0.09)],
        "top_3_negative": [
            ("user_interaction", -0.12),
            ("attack_complexity", -0.07),
            ("scope", -0.03),
        ],
    }
    return pred


def _make_mock_retriever():
    ret = MagicMock()
    ret.retrieve_for_cve.return_value = [
        {
            "content": "Use parameterized queries.",
            "source_file": "sql_injection_remediation.md",
            "chunk_index": 0,
            "similarity_score": 0.12,
        }
    ]
    ret.retrieve.return_value = ret.retrieve_for_cve.return_value
    return ret


_SAMPLE_CVE = {
    "cve_id": "CVE-2024-21762",
    "description": "Out-of-bounds write in FortiOS allows unauthenticated RCE.",
    "cvss_v3_score": 9.8,
    "attack_vector": "NETWORK",
    "attack_complexity": "LOW",
    "privileges_required": "NONE",
    "user_interaction": "NONE",
    "scope": "UNCHANGED",
    "confidentiality_impact": "HIGH",
    "integrity_impact": "HIGH",
    "availability_impact": "HIGH",
    "cwe_ids": ["CWE-787"],
    "references": ["https://github.com/researcher/poc"],
    "has_exploit_ref": True,
    "published_date": "2024-02-09T00:00:00.000",
    "last_modified_date": "2024-03-15T12:34:56.000",
    "affected_products": ["cpe:2.3:o:fortinet:fortios:7.4.0:*:*:*:*:*:*:*"],
}


@pytest.fixture()
def client_no_model():
    """TestClient with no model or index loaded."""
    import api.dependencies as deps
    from api.main import app

    deps.app_state.model_loaded = False
    deps.app_state.index_loaded = False
    deps.app_state.predictor = None
    deps.app_state.retriever = None

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture()
def client_ready():
    """TestClient with mocked predictor and retriever loaded."""
    import api.dependencies as deps
    from api.main import app

    mock_predictor = _make_mock_predictor()
    mock_retriever = _make_mock_retriever()

    deps.app_state.model_loaded = True
    deps.app_state.index_loaded = True
    deps.app_state.predictor = mock_predictor
    deps.app_state.retriever = mock_retriever

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # cleanup
    deps.app_state.model_loaded = False
    deps.app_state.index_loaded = False
    deps.app_state.predictor = None
    deps.app_state.retriever = None


# ---------------------------------------------------------------------------
# test_health_endpoint_returns_status
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_status(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "index_loaded" in body
    assert "version" in body


def test_health_model_loaded_reflects_state(client_ready):
    response = client_ready.get("/health")
    body = response.json()
    assert body["model_loaded"] is True
    assert body["index_loaded"] is True


# ---------------------------------------------------------------------------
# test_predict_endpoint_returns_severity
# ---------------------------------------------------------------------------


def test_predict_endpoint_returns_severity(client_ready):
    """POST /predict returns a SeverityResponse with all required fields."""
    with patch("api.routes.predict._fetch_single_cve", return_value=_SAMPLE_CVE):
        response = client_ready.post("/predict", json={"cve_id": "CVE-2024-21762"})

    assert response.status_code == 200
    body = response.json()
    assert body["cve_id"] == "CVE-2024-21762"
    assert "exploit_probability" in body
    assert body["risk_level"] in ("HIGH", "MEDIUM", "LOW")
    assert "shap_explanation" in body
    assert "top_positive_features" in body["shap_explanation"]
    assert "top_negative_features" in body["shap_explanation"]


def test_predict_with_precomputed_features(client_ready):
    """POST /predict with features dict skips NVD fetch."""
    features = {f"f{i}": 0.0 for i in range(30)}
    features["cvss_v3_score"] = 9.8

    response = client_ready.post(
        "/predict",
        json={"cve_id": "CVE-2024-21762", "features": features},
    )
    assert response.status_code == 200
    assert response.json()["cve_id"] == "CVE-2024-21762"


# ---------------------------------------------------------------------------
# test_playbook_endpoint_returns_playbook
# ---------------------------------------------------------------------------


def test_playbook_endpoint_returns_playbook(client_ready):
    """POST /playbook returns a PlaybookResponse with sections and sources."""
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "## 1. Executive Summary\nCritical RCE vulnerability.\n\n"
        "## 2. Risk Assessment\nCVSS 9.8 critical.\n\n"
        "## 3. Immediate Actions\n1. Patch immediately.\n\n"
        "## 4. Detection Rules\nMonitor process trees.\n\n"
        "## 5. Monitoring Recommendations\nEnable enhanced logging."
    )

    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
        patch("rag.playbook_generator.ChatGroq") as mock_groq,
    ):
        mock_groq.return_value.invoke.return_value = mock_llm_response
        response = client_ready.post(
            "/playbook",
            json={
                "cve_id": "CVE-2024-21762",
                "description": "Out-of-bounds write allows RCE",
                "severity": "CRITICAL",
                "cwe": "CWE-787",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["cve_id"] == "CVE-2024-21762"
    assert isinstance(body["playbook"], dict)
    assert len(body["playbook"]) > 0
    assert isinstance(body["sources"], list)


# ---------------------------------------------------------------------------
# test_analyze_endpoint_returns_full_response
# ---------------------------------------------------------------------------


def test_analyze_endpoint_returns_full_response(client_ready):
    """POST /analyze returns AnalyzeResponse combining severity + playbook."""
    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "## 1. Executive Summary\nCritical issue.\n\n"
        "## 2. Risk Assessment\nHigh risk.\n\n"
        "## 3. Immediate Actions\n1. Apply patch.\n\n"
        "## 4. Detection Rules\nMonitor logs.\n\n"
        "## 5. Monitoring Recommendations\nWatch for anomalies."
    )

    with (
        patch("api.routes.analyze._fetch_single_cve", return_value=_SAMPLE_CVE),
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
        patch("rag.playbook_generator.ChatGroq") as mock_groq,
    ):
        mock_groq.return_value.invoke.return_value = mock_llm_response
        response = client_ready.post("/analyze", json={"cve_id": "CVE-2024-21762"})

    assert response.status_code == 200
    body = response.json()
    assert body["cve_id"] == "CVE-2024-21762"
    assert "severity" in body
    assert "playbook" in body
    assert body["severity"]["risk_level"] in ("HIGH", "MEDIUM", "LOW")
    assert "exploit_probability" in body["severity"]
    assert isinstance(body["playbook"]["playbook"], dict)


# ---------------------------------------------------------------------------
# test_analyze_returns_503_when_model_not_loaded
# ---------------------------------------------------------------------------


def test_analyze_returns_503_when_model_not_loaded(client_no_model):
    """POST /analyze returns 503 with clear message when model is not trained."""
    response = client_no_model.post("/analyze", json={"cve_id": "CVE-2024-21762"})
    assert response.status_code == 503
    assert "Model not trained" in response.json()["detail"]


def test_predict_returns_503_when_model_not_loaded(client_no_model):
    """POST /predict returns 503 when model is not loaded."""
    response = client_no_model.post("/predict", json={"cve_id": "CVE-2024-21762"})
    assert response.status_code == 503


def test_playbook_returns_503_when_index_not_loaded(client_no_model):
    """POST /playbook returns 503 when FAISS index is not loaded."""
    response = client_no_model.post(
        "/playbook",
        json={
            "cve_id": "CVE-2024-21762",
            "description": "RCE vulnerability",
            "severity": "CRITICAL",
            "cwe": "CWE-787",
        },
    )
    assert response.status_code == 503
