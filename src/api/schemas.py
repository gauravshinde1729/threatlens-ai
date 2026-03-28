"""Pydantic request/response schemas for the ThreatLens AI REST API."""

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    cve_id: str = Field(..., description="CVE identifier, e.g. CVE-2024-21762")


class PredictRequest(BaseModel):
    cve_id: str = Field(..., description="CVE identifier")
    features: dict[str, Any] | None = Field(
        default=None,
        description="Pre-computed feature dict; if omitted the CVE is fetched from NVD",
    )


class PlaybookRequest(BaseModel):
    cve_id: str = Field(..., description="CVE identifier")
    description: str = Field(..., description="Vulnerability description text")
    severity: str = Field(..., description="CVSS severity label (CRITICAL/HIGH/MEDIUM/LOW)")
    cwe: str = Field(..., description="Primary CWE identifier, e.g. CWE-89")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ShapExplanation(BaseModel):
    top_positive_features: list[tuple[str, float]] = Field(
        description="Features pushing toward exploitation risk (name, shap_value)"
    )
    top_negative_features: list[tuple[str, float]] = Field(
        description="Features pushing against exploitation risk (name, shap_value)"
    )


class SeverityResponse(BaseModel):
    cve_id: str
    cvss_score: float | None
    exploit_probability: float
    risk_level: str = Field(description="HIGH / MEDIUM / LOW derived from exploit_probability")
    shap_explanation: ShapExplanation


class PlaybookResponse(BaseModel):
    cve_id: str
    playbook: dict[str, str] = Field(description="Playbook sections keyed by section name")
    sources: list[str] = Field(description="Knowledge base source files used")


class AnalyzeResponse(BaseModel):
    cve_id: str
    severity: SeverityResponse
    playbook: PlaybookResponse


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    error_type: str
    detail: str
