"""LLM-powered remediation playbook generator using Groq + LangChain."""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from rag.prompts.remediation import REMEDIATION_PROMPT_TEMPLATE

load_dotenv()

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_FALLBACK_MODEL = "mixtral-8x7b-32768"


class PlaybookGenerator:
    """Generate structured remediation playbooks using a Groq-hosted LLM."""

    def __init__(self, retriever: Any, model: str = _GROQ_MODEL) -> None:
        self._retriever = retriever
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it in your .env file or shell environment."
            )
        self._llm = ChatGroq(model=model, api_key=api_key)

    def generate(
        self,
        cve_data: dict[str, Any],
        ml_prediction: dict[str, Any],
        retrieved_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate a remediation playbook for a CVE.

        Args:
            cve_data: Parsed CVE dict from NVDClient.
            ml_prediction: Dict with at minimum exploit_probability and
                predicted_label keys (from SeverityPredictor).
            retrieved_docs: Relevant knowledge base chunks from SecurityRetriever.

        Returns:
            Dict with keys:
                - cve_id (str)
                - playbook (str): full LLM-generated playbook text
                - sources (list[str]): source filenames used as context
                - model (str): LLM model identifier used
        """
        cve_id = cve_data.get("cve_id", "UNKNOWN")
        logger.info("Generating playbook for %s", cve_id)

        cve_details = self._format_cve_details(cve_data)
        ml_text = self._format_ml_prediction(ml_prediction)
        context_text = self._format_retrieved_docs(retrieved_docs)

        prompt = REMEDIATION_PROMPT_TEMPLATE.format(
            cve_details=cve_details,
            ml_prediction=ml_text,
            retrieved_context=context_text,
        )

        response = self._llm.invoke([HumanMessage(content=prompt)])
        playbook_text = response.content

        sources = list({doc["source_file"] for doc in retrieved_docs})
        logger.info("Playbook generated for %s (%d chars)", cve_id, len(playbook_text))

        return {
            "cve_id": cve_id,
            "playbook": playbook_text,
            "sources": sources,
            "model": _GROQ_MODEL,
        }

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_cve_details(cve: dict[str, Any]) -> str:
        lines = [
            f"CVE ID: {cve.get('cve_id', 'N/A')}",
            f"Description: {cve.get('description', 'N/A')}",
            f"CVSS v3 Score: {cve.get('cvss_v3_score', 'N/A')}",
            f"Attack Vector: {cve.get('attack_vector', 'N/A')}",
            f"Attack Complexity: {cve.get('attack_complexity', 'N/A')}",
            f"Privileges Required: {cve.get('privileges_required', 'N/A')}",
            f"User Interaction: {cve.get('user_interaction', 'N/A')}",
            f"CWE IDs: {', '.join(cve.get('cwe_ids') or ['N/A'])}",
            f"Affected Products: {', '.join((cve.get('affected_products') or [])[:5]) or 'N/A'}",
            f"Published: {cve.get('published_date', 'N/A')}",
            f"Has Exploit Reference: {cve.get('has_exploit_ref', False)}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_ml_prediction(prediction: dict[str, Any]) -> str:
        prob = prediction.get("exploit_probability")
        prob_str = f"{prob:.1%}" if prob is not None else "N/A"
        label = prediction.get("predicted_label", "N/A")
        confidence = prediction.get("confidence", "N/A")
        return (
            f"Exploit Probability: {prob_str}\n"
            f"Predicted Label: {label}\n"
            f"Model Confidence: {confidence}"
        )

    @staticmethod
    def _format_retrieved_docs(docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "No relevant documentation found."
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source_file", "unknown")
            content = doc.get("content", "").strip()
            parts.append(f"[Source {i}: {source}]\n{content}")
        return "\n\n---\n\n".join(parts)
