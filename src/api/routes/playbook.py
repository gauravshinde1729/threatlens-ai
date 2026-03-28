"""POST /playbook — RAG-only playbook generation from CVE details."""

import logging
import re

from fastapi import APIRouter, HTTPException

from api.dependencies import app_state
from api.schemas import PlaybookRequest, PlaybookResponse
from rag.playbook_generator import PlaybookGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

# Section headers expected in the LLM output
_SECTION_PATTERNS = {
    "executive_summary": re.compile(r"(?i)executive summary"),
    "risk_assessment": re.compile(r"(?i)risk assessment"),
    "immediate_actions": re.compile(r"(?i)immediate actions"),
    "detection_rules": re.compile(r"(?i)detection rules"),
    "monitoring_recommendations": re.compile(r"(?i)monitoring recommendations"),
}


@router.post("/playbook", response_model=PlaybookResponse)
async def generate_playbook(request: PlaybookRequest) -> PlaybookResponse:
    """Generate a remediation playbook for a CVE using RAG + LLM.

    Accepts CVE details directly (no NVD fetch required) and returns a
    structured playbook with five sections.
    """
    if not app_state.index_loaded or app_state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run training pipeline first.",
        )

    cve_data = {
        "cve_id": request.cve_id,
        "description": request.description,
        "cwe_ids": [request.cwe] if request.cwe else [],
        "attack_vector": "NETWORK",
        "cvss_v3_score": None,
        "has_exploit_ref": False,
        "affected_products": [],
        "references": [],
    }

    docs = app_state.retriever.retrieve_for_cve(cve_data, top_k=5)

    try:
        generator = PlaybookGenerator(app_state.retriever)
    except EnvironmentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    ml_placeholder = {
        "exploit_probability": None,
        "predicted_label": "N/A (predict endpoint for ML score)",
        "confidence": "N/A",
    }

    result = generator.generate(cve_data, ml_placeholder, docs)
    playbook_text: str = result["playbook"]
    sections = _parse_sections(playbook_text)

    logger.info("Playbook generated for %s (%d sections)", request.cve_id, len(sections))

    return PlaybookResponse(
        cve_id=request.cve_id,
        playbook=sections,
        sources=result["sources"],
    )


def _parse_sections(text: str) -> dict[str, str]:
    """Split LLM output into named sections by header markers."""
    lines = text.split("\n")
    sections: dict[str, str] = {}
    current_key: str | None = None
    buffer: list[str] = []

    for line in lines:
        matched_key = None
        for key, pattern in _SECTION_PATTERNS.items():
            if pattern.search(line):
                matched_key = key
                break

        if matched_key:
            if current_key and buffer:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = matched_key
            buffer = []
        elif current_key:
            buffer.append(line)

    if current_key and buffer:
        sections[current_key] = "\n".join(buffer).strip()

    # If parsing failed (unexpected format), return raw text as single section
    if not sections:
        sections["full_playbook"] = text.strip()

    return sections
