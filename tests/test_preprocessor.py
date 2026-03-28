"""Tests for src/data/preprocessor.py."""

import json
from pathlib import Path

import pandas as pd
import pytest

from data.nvd_client import NVDClient
from data.preprocessor import (
    _REQUIRED_COLUMNS,
    TOP_CWES,
    CVEPreprocessor,
    _compute_is_exploited,
)

SAMPLE_CVES_PATH = Path(__file__).parent / "test_data" / "sample_cves.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_nvd_entries() -> list[dict]:
    return json.loads(SAMPLE_CVES_PATH.read_text())


@pytest.fixture()
def parsed_cves(raw_nvd_entries) -> list[dict]:
    """Parse sample NVD entries through NVDClient into flat CVE dicts."""
    client = NVDClient()
    return [client._parse_cve(item["cve"]) for item in raw_nvd_entries]


@pytest.fixture()
def preprocessor() -> CVEPreprocessor:
    return CVEPreprocessor()


@pytest.fixture()
def feature_df(preprocessor, parsed_cves) -> pd.DataFrame:
    return preprocessor.transform(parsed_cves)


# ---------------------------------------------------------------------------
# test_transform_returns_dataframe_with_correct_columns
# ---------------------------------------------------------------------------


def test_transform_returns_dataframe_with_correct_columns(feature_df, parsed_cves):
    """transform() returns a DataFrame with exactly the required columns and correct row count."""
    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_df) == len(parsed_cves)
    for col in _REQUIRED_COLUMNS:
        assert col in feature_df.columns, f"Missing column: {col}"


def test_transform_empty_list_returns_empty_dataframe(preprocessor):
    """transform([]) returns an empty DataFrame that still has all required columns."""
    df = preprocessor.transform([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    for col in _REQUIRED_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# test_transform_handles_missing_cvss_gracefully
# ---------------------------------------------------------------------------


def test_transform_handles_missing_cvss_gracefully(preprocessor, parsed_cves):
    """CVE with no CVSS data produces 0-filled numeric/ordinal fields without raising."""
    no_cvss = next(c for c in parsed_cves if c["cve_id"] == "CVE-2024-45678")
    # Confirm fixture has no CVSS
    assert no_cvss["cvss_v3_score"] is None

    df = preprocessor.transform([no_cvss])
    row = df.iloc[0]

    assert row["cvss_v3_score"] == 0.0
    assert row["attack_vector"] == 0
    assert row["attack_complexity"] == 0
    assert row["privileges_required"] == 0
    assert row["confidentiality_impact"] == 0
    assert row["integrity_impact"] == 0
    assert row["availability_impact"] == 0


# ---------------------------------------------------------------------------
# test_keyword_extraction_is_case_insensitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "description,expected_flags",
    [
        (
            "This allows Remote Code Execution via a crafted request.",
            {"has_keyword_rce": 1},
        ),
        (
            "Vulnerability enables SQL Injection in the login form.",
            {"has_keyword_sqli": 1},
        ),
        (
            "Stored XSS via unsanitised input in the comment field.",
            {"has_keyword_xss": 1},
        ),
        (
            "Auth Bypass due to missing authentication check.",
            {"has_keyword_auth_bypass": 1},
        ),
        (
            "Buffer Overflow in the parsing routine causes crash.",
            {"has_keyword_buffer_overflow": 1},
        ),
        (
            "Allows PRIVILEGE ESCALATION to root.",
            {"has_keyword_privilege_escalation": 1},
        ),
        (
            # Abbreviation forms (upper and lower)
            "Unauthenticated RCE in the web interface.",
            {"has_keyword_rce": 1},
        ),
        (
            "Cross-Site Scripting in search results.",
            {"has_keyword_xss": 1},
        ),
        (
            # No keywords present
            "An information disclosure in the debug endpoint.",
            {
                "has_keyword_rce": 0,
                "has_keyword_sqli": 0,
                "has_keyword_xss": 0,
                "has_keyword_auth_bypass": 0,
                "has_keyword_buffer_overflow": 0,
                "has_keyword_privilege_escalation": 0,
            },
        ),
    ],
)
def test_keyword_extraction_is_case_insensitive(preprocessor, description, expected_flags):
    cve = _minimal_cve(description=description)
    df = preprocessor.transform([cve])
    row = df.iloc[0]
    for flag, expected_val in expected_flags.items():
        assert row[flag] == expected_val, f"{flag}: expected {expected_val}, got {row[flag]}"


# ---------------------------------------------------------------------------
# test_cwe_one_hot_encoding
# ---------------------------------------------------------------------------


def test_cwe_one_hot_encoding(preprocessor):
    """Top CWEs get individual columns; unknown CWEs set cwe_other=1."""
    cve_top = _minimal_cve(cwe_ids=["CWE-79", "CWE-787"])
    cve_other = _minimal_cve(cwe_ids=["CWE-999"])
    cve_mixed = _minimal_cve(cwe_ids=["CWE-89", "CWE-1234"])
    cve_none = _minimal_cve(cwe_ids=[])

    df = preprocessor.transform([cve_top, cve_other, cve_mixed, cve_none])

    top_row = df.iloc[0]
    assert top_row["cwe_79"] == 1
    assert top_row["cwe_787"] == 1
    assert top_row["cwe_other"] == 0
    # Other top-10 CWEs not in the list must be 0
    assert top_row["cwe_89"] == 0

    other_row = df.iloc[1]
    assert other_row["cwe_other"] == 1
    for cwe in TOP_CWES:
        col = f"cwe_{cwe.split('-')[1].lower()}"
        assert other_row[col] == 0

    mixed_row = df.iloc[2]
    assert mixed_row["cwe_89"] == 1
    assert mixed_row["cwe_other"] == 1

    none_row = df.iloc[3]
    assert none_row["cwe_other"] == 0
    for cwe in TOP_CWES:
        assert none_row[f"cwe_{cwe.split('-')[1].lower()}"] == 0


# ---------------------------------------------------------------------------
# test_is_exploited_target_variable_logic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "has_exploit_ref,references,expected",
    [
        # has_exploit_ref alone triggers True
        (True, [], True),
        # CISA KEV URL triggers True
        (False, ["https://www.cisa.gov/known-exploited-vulnerabilities-catalog"], True),
        # kevchecker URL triggers True
        (False, ["https://kevchecker.com/check/CVE-2024-0001"], True),
        # Both signals present
        (True, ["https://www.cisa.gov/known-exploited-vulnerabilities-catalog"], True),
        # No signals → False
        (False, ["https://nvd.nist.gov/vuln/detail/CVE-2024-0001"], False),
        (False, [], False),
    ],
)
def test_is_exploited_target_variable_logic(preprocessor, has_exploit_ref, references, expected):
    cve = _minimal_cve(has_exploit_ref=has_exploit_ref, references=references)
    df = preprocessor.transform([cve])
    assert df.iloc[0]["is_exploited"] == int(expected)


def test_is_exploited_via_compute_helper():
    """_compute_is_exploited is consistent with the DataFrame column."""
    cve_kev = _minimal_cve(
        references=["https://www.cisa.gov/known-exploited-vulnerabilities-catalog"]
    )
    assert _compute_is_exploited(cve_kev) is True

    cve_clean = _minimal_cve(references=["https://example.com/advisory"])
    assert _compute_is_exploited(cve_clean) is False


# ---------------------------------------------------------------------------
# test_ordinal_encoding_of_cvss_components
# ---------------------------------------------------------------------------


def test_ordinal_encoding_of_cvss_components(preprocessor):
    """CVSS categorical values are mapped to correct ordinal integers."""
    cve = _minimal_cve(
        attack_vector="NETWORK",
        attack_complexity="LOW",
        privileges_required="NONE",
        user_interaction="NONE",
        scope="CHANGED",
        confidentiality_impact="HIGH",
        integrity_impact="LOW",
        availability_impact="NONE",
    )
    df = preprocessor.transform([cve])
    row = df.iloc[0]

    assert row["attack_vector"] == 3  # NETWORK is highest
    assert row["attack_complexity"] == 1  # LOW (less complex = higher risk)
    assert row["privileges_required"] == 2  # NONE = most permissive
    assert row["user_interaction"] == 1  # NONE = no interaction needed
    assert row["scope"] == 1  # CHANGED
    assert row["confidentiality_impact"] == 2  # HIGH
    assert row["integrity_impact"] == 1  # LOW
    assert row["availability_impact"] == 0  # NONE


def test_ordinal_encoding_unknown_values(preprocessor):
    """Missing/None categorical fields fall back to 0, not an error."""
    cve = _minimal_cve(
        attack_vector=None,
        attack_complexity=None,
        privileges_required=None,
    )
    df = preprocessor.transform([cve])
    row = df.iloc[0]
    assert row["attack_vector"] == 0
    assert row["attack_complexity"] == 0
    assert row["privileges_required"] == 0


# ---------------------------------------------------------------------------
# save_features / load_features round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_features_round_trip(preprocessor, parsed_cves, tmp_path):
    """save_features + load_features preserves DataFrame content via CSV."""
    df = preprocessor.transform(parsed_cves)
    path = tmp_path / "features.csv"
    preprocessor.save_features(df, path)
    loaded = preprocessor.load_features(path)
    pd.testing.assert_frame_equal(df, loaded, check_dtype=False)


# ---------------------------------------------------------------------------
# Numeric feature sanity checks
# ---------------------------------------------------------------------------


def test_numeric_features_are_populated(feature_df):
    """description_length, reference_count, affected_product_count, days_since_publication > 0."""
    rce_row = feature_df[feature_df["cve_id"] == "CVE-2024-21762"].iloc[0]
    assert rce_row["description_length"] > 0
    assert rce_row["reference_count"] == 3
    assert rce_row["affected_product_count"] == 2
    assert rce_row["days_since_publication"] > 0


def test_days_since_publication_positive(feature_df):
    """All sample CVEs (published in 2024) have positive days_since_publication."""
    assert (feature_df["days_since_publication"] > 0).all()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _minimal_cve(
    cve_id: str = "CVE-2024-00001",
    description: str = "A test vulnerability.",
    cvss_v3_score: float | None = None,
    references: list[str] | None = None,
    cwe_ids: list[str] | None = None,
    has_exploit_ref: bool = False,
    published_date: str = "2024-01-01T00:00:00.000",
    affected_products: list[str] | None = None,
    attack_vector: str | None = None,
    attack_complexity: str | None = None,
    privileges_required: str | None = None,
    user_interaction: str | None = None,
    scope: str | None = None,
    confidentiality_impact: str | None = None,
    integrity_impact: str | None = None,
    availability_impact: str | None = None,
) -> dict:
    return {
        "cve_id": cve_id,
        "description": description,
        "cvss_v3_score": cvss_v3_score,
        "references": references or [],
        "cwe_ids": cwe_ids or [],
        "has_exploit_ref": has_exploit_ref,
        "published_date": published_date,
        "affected_products": affected_products or [],
        "attack_vector": attack_vector,
        "attack_complexity": attack_complexity,
        "privileges_required": privileges_required,
        "user_interaction": user_interaction,
        "scope": scope,
        "confidentiality_impact": confidentiality_impact,
        "integrity_impact": integrity_impact,
        "availability_impact": availability_impact,
    }
