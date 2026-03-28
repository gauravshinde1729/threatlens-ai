"""Retrieves relevant security documentation chunks for a given query or CVE."""

import logging
from typing import Any

from rag.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

# Map CWE IDs to human-readable domain terms that improve retrieval recall
_CWE_QUERY_HINTS: dict[str, str] = {
    "CWE-77": "command injection remediation",
    "CWE-78": "command injection os command remediation",
    "CWE-79": "cross-site scripting xss prevention output encoding",
    "CWE-89": "sql injection prepared statements parameterized query",
    "CWE-119": "buffer overflow memory corruption bounds checking",
    "CWE-125": "out-of-bounds read memory safety",
    "CWE-190": "integer overflow memory corruption",
    "CWE-200": "information disclosure log monitoring detection",
    "CWE-269": "privilege escalation least privilege rbac",
    "CWE-287": "authentication bypass session management mfa",
    "CWE-416": "use after free memory corruption heap",
    "CWE-476": "null pointer dereference memory safety",
    "CWE-787": "buffer overflow out-of-bounds write memory corruption",
}

# Attack vector terms → retrieval hint
_ATTACK_VECTOR_HINTS: dict[str, str] = {
    "NETWORK": "network segmentation firewall",
    "ADJACENT": "network segmentation micro-segmentation",
    "LOCAL": "privilege escalation local access",
    "PHYSICAL": "physical access hardening",
}


class SecurityRetriever:
    """Retrieve relevant security documentation chunks from the knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase, top_k: int = 5) -> None:
        self._kb = knowledge_base
        self._default_top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve the most relevant chunks for a free-text query.

        Args:
            query: Natural language query string.
            top_k: Number of results; defaults to the instance's default_top_k.

        Returns:
            List of result dicts: content, source_file, chunk_index,
            similarity_score.
        """
        k = top_k if top_k is not None else self._default_top_k
        logger.debug("Retrieving top-%d for query: %r", k, query[:80])
        return self._kb.search(query, top_k=k)

    def retrieve_for_cve(
        self, cve_data: dict[str, Any], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Build a semantically rich query from CVE metadata and retrieve chunks.

        The query combines CWE domain terms, key description words, and
        attack-vector hints for higher-relevance retrieval than a raw
        description search.

        Args:
            cve_data: Parsed CVE dict as produced by NVDClient.
            top_k: Number of results to return.

        Returns:
            List of result dicts (same schema as retrieve()).
        """
        query_parts: list[str] = []

        # CWE-specific remediation hints
        for cwe in cve_data.get("cwe_ids") or []:
            hint = _CWE_QUERY_HINTS.get(cwe)
            if hint:
                query_parts.append(hint)

        # Description keywords (first 200 chars to focus on the vulnerability type)
        description = (cve_data.get("description") or "").strip()
        if description:
            query_parts.append(description[:200])

        # Attack vector context
        av = (cve_data.get("attack_vector") or "").upper()
        av_hint = _ATTACK_VECTOR_HINTS.get(av)
        if av_hint:
            query_parts.append(av_hint)

        # Fallback: if we have nothing useful, use cve_id as a last resort
        if not query_parts:
            query_parts.append(cve_data.get("cve_id", "vulnerability remediation"))

        query = " ".join(query_parts)
        logger.info(
            "CVE %s smart query (%d chars): %r",
            cve_data.get("cve_id", "unknown"),
            len(query),
            query[:120],
        )
        return self.retrieve(query, top_k=top_k)
