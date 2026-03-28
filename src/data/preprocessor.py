"""Transform parsed CVE dicts into ML-ready feature DataFrames."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ordinal encodings (higher value = more exploitable / higher impact)
# ---------------------------------------------------------------------------

_ATTACK_VECTOR_ORD: dict[str, int] = {
    "PHYSICAL": 0,
    "LOCAL": 1,
    "ADJACENT": 2,
    "ADJACENT_NETWORK": 2,
    "NETWORK": 3,
    "UNKNOWN": 0,
}
_ATTACK_COMPLEXITY_ORD: dict[str, int] = {"HIGH": 0, "LOW": 1, "UNKNOWN": 0}
_PRIVILEGES_REQUIRED_ORD: dict[str, int] = {"HIGH": 0, "LOW": 1, "NONE": 2, "UNKNOWN": 0}
_USER_INTERACTION_ORD: dict[str, int] = {"REQUIRED": 0, "NONE": 1, "UNKNOWN": 0}
_SCOPE_ORD: dict[str, int] = {"UNCHANGED": 0, "CHANGED": 1, "UNKNOWN": 0}
_IMPACT_ORD: dict[str, int] = {"NONE": 0, "LOW": 1, "HIGH": 2, "UNKNOWN": 0}

# ---------------------------------------------------------------------------
# CWE one-hot config
# ---------------------------------------------------------------------------

TOP_CWES: list[str] = [
    "CWE-79",
    "CWE-89",
    "CWE-787",
    "CWE-416",
    "CWE-78",
    "CWE-20",
    "CWE-125",
    "CWE-476",
    "CWE-190",
    "CWE-119",
]
_TOP_CWE_SET: frozenset[str] = frozenset(TOP_CWES)

# ---------------------------------------------------------------------------
# Keyword patterns (compiled once)
# ---------------------------------------------------------------------------

_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "has_keyword_rce": re.compile(r"remote code execution|(?<!\w)rce(?!\w)", re.IGNORECASE),
    "has_keyword_sqli": re.compile(r"sql injection|(?<!\w)sqli(?!\w)", re.IGNORECASE),
    "has_keyword_xss": re.compile(r"cross-site scripting|(?<!\w)xss(?!\w)", re.IGNORECASE),
    "has_keyword_auth_bypass": re.compile(r"authentication bypass|auth bypass", re.IGNORECASE),
    "has_keyword_buffer_overflow": re.compile(r"buffer overflow", re.IGNORECASE),
    "has_keyword_privilege_escalation": re.compile(r"privilege escalation", re.IGNORECASE),
}

# Columns the DataFrame must always contain, in order
_REQUIRED_COLUMNS: list[str] = (
    [
        "cve_id",
        "cvss_v3_score",
        "description_length",
        "reference_count",
        "affected_product_count",
        "days_since_publication",
        "attack_vector",
        "attack_complexity",
        "privileges_required",
        "user_interaction",
        "scope",
        "confidentiality_impact",
        "integrity_impact",
        "availability_impact",
    ]
    + list(_KEYWORD_PATTERNS.keys())
    + ["has_exploit_ref"]
    + [f"cwe_{cwe.split('-')[1].lower()}" for cwe in TOP_CWES]
    + ["cwe_other", "is_exploited"]
)


class CVEPreprocessor:
    """Transform parsed CVE dicts (output of NVDClient) into a feature DataFrame."""

    def transform(self, cves: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of parsed CVE dicts into an ML-ready DataFrame.

        Args:
            cves: List of CVE dicts as returned by NVDClient.fetch_cves().

        Returns:
            DataFrame with one row per CVE and fully encoded features.
        """
        logger.info("Transforming %d CVEs into feature DataFrame", len(cves))
        rows = [self._build_row(cve) for cve in cves]
        df = pd.DataFrame(rows)
        # Ensure all required columns exist (handles empty input)
        for col in _REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = 0
        logger.info("Feature DataFrame shape: %s", df.shape)
        return df[_REQUIRED_COLUMNS]

    @staticmethod
    def save_features(df: pd.DataFrame, path: str | Path) -> None:
        """Persist a feature DataFrame to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved features to %s", path)

    @staticmethod
    def load_features(path: str | Path) -> pd.DataFrame:
        """Load a feature DataFrame from a CSV file."""
        path = Path(path)
        df = pd.read_csv(path)
        logger.info("Loaded features from %s (%d rows)", path, len(df))
        return df

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    def _build_row(self, cve: dict[str, Any]) -> dict[str, Any]:
        description: str = cve.get("description") or ""
        references: list[str] = cve.get("references") or []
        cwe_ids: list[str] = cve.get("cwe_ids") or []
        score = cve.get("cvss_v3_score")

        row: dict[str, Any] = {
            "cve_id": cve.get("cve_id", ""),
            # Numeric features
            "cvss_v3_score": float(score) if score is not None else 0.0,
            "description_length": len(description),
            "reference_count": len(references),
            "affected_product_count": len(cve.get("affected_products") or []),
            "days_since_publication": _days_since(cve.get("published_date")),
            # Ordinal-encoded CVSS components
            "attack_vector": _encode(cve, "attack_vector", _ATTACK_VECTOR_ORD),
            "attack_complexity": _encode(cve, "attack_complexity", _ATTACK_COMPLEXITY_ORD),
            "privileges_required": _encode(cve, "privileges_required", _PRIVILEGES_REQUIRED_ORD),
            "user_interaction": _encode(cve, "user_interaction", _USER_INTERACTION_ORD),
            "scope": _encode(cve, "scope", _SCOPE_ORD),
            "confidentiality_impact": _encode(cve, "confidentiality_impact", _IMPACT_ORD),
            "integrity_impact": _encode(cve, "integrity_impact", _IMPACT_ORD),
            "availability_impact": _encode(cve, "availability_impact", _IMPACT_ORD),
            # Boolean keyword flags
            **{name: int(pat.search(description) is not None) for name, pat in _KEYWORD_PATTERNS.items()},
            # Exploit reference flag (already parsed by NVDClient)
            "has_exploit_ref": int(bool(cve.get("has_exploit_ref", False))),
            # CWE one-hot columns
            **_encode_cwes(cwe_ids),
            # Target variable
            "is_exploited": int(_compute_is_exploited(cve)),
        }
        return row


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _encode(cve: dict[str, Any], field: str, mapping: dict[str, int]) -> int:
    raw = (cve.get(field) or "UNKNOWN").upper()
    return mapping.get(raw, mapping.get("UNKNOWN", 0))


def _encode_cwes(cwe_ids: list[str]) -> dict[str, int]:
    result: dict[str, int] = {f"cwe_{cwe.split('-')[1].lower()}": 0 for cwe in TOP_CWES}
    result["cwe_other"] = 0
    has_other = False
    for cwe in cwe_ids:
        if cwe in _TOP_CWE_SET:
            col = f"cwe_{cwe.split('-')[1].lower()}"
            result[col] = 1
        elif cwe:
            has_other = True
    result["cwe_other"] = int(has_other)
    return result


def _days_since(date_str: str | None) -> int:
    if not date_str:
        return 0
    try:
        # NVD dates: "2024-02-09T00:00:00.000" — no timezone suffix
        pub = datetime.fromisoformat(date_str.rstrip("Z").split(".")[0])
        pub = pub.replace(tzinfo=timezone.utc)
        delta = datetime.now(tz=timezone.utc) - pub
        return max(0, delta.days)
    except (ValueError, TypeError):
        return 0


def _compute_is_exploited(cve: dict[str, Any]) -> bool:
    if cve.get("has_exploit_ref", False):
        return True
    for url in cve.get("references") or []:
        url_lower = url.lower()
        if "cisa.gov/known-exploited" in url_lower or "kevchecker" in url_lower:
            return True
    return False
