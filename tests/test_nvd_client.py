"""Tests for src/data/nvd_client.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from data.nvd_client import NVDClient, NVDRateLimiter

SAMPLE_CVES_PATH = Path(__file__).parent / "test_data" / "sample_cves.json"


@pytest.fixture()
def sample_cves() -> list[dict]:
    return json.loads(SAMPLE_CVES_PATH.read_text())


def _make_nvd_response(vulnerabilities: list[dict], total: int | None = None) -> dict:
    """Build a minimal NVD API response envelope."""
    return {
        "totalResults": total if total is not None else len(vulnerabilities),
        "resultsPerPage": len(vulnerabilities),
        "startIndex": 0,
        "vulnerabilities": vulnerabilities,
    }


def _mock_httpx_response(payload: dict, status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=response
        )
    return response


# ---------------------------------------------------------------------------
# test_fetch_cves_returns_parsed_data
# ---------------------------------------------------------------------------


def test_fetch_cves_returns_parsed_data(sample_cves):
    """Parsed CVEs contain all expected fields with correct values.

    CVE-2024-45678 has no cvssMetricV31 and is skipped by the filter,
    so only the two CVEs with CVSS v3.1 data are returned.
    """
    payload = _make_nvd_response(sample_cves)
    client = NVDClient(cache_path=None)

    with patch.object(client, "_fetch_page", return_value=payload):
        results = client.fetch_cves(max_results=10)

    # Only the 2 CVEs that have cvssMetricV31 are returned
    assert len(results) == 2

    rce = next(r for r in results if r["cve_id"] == "CVE-2024-21762")
    assert rce["cvss_v3_score"] == 9.8
    assert rce["attack_vector"] == "NETWORK"
    assert rce["attack_complexity"] == "LOW"
    assert rce["privileges_required"] == "NONE"
    assert rce["user_interaction"] == "NONE"
    assert rce["scope"] == "UNCHANGED"
    assert rce["confidentiality_impact"] == "HIGH"
    assert rce["integrity_impact"] == "HIGH"
    assert rce["availability_impact"] == "HIGH"
    assert rce["cwe_ids"] == ["CWE-787"]
    assert "FortiOS" in rce["description"]
    assert rce["published_date"] == "2024-02-09T00:00:00.000"
    assert rce["last_modified_date"] == "2024-03-15T12:34:56.000"
    assert len(rce["affected_products"]) == 2
    assert rce["cvss_v3_vector"].startswith("CVSS:3.1")


# ---------------------------------------------------------------------------
# test_fetch_cves_skips_missing_cvss
# ---------------------------------------------------------------------------


def test_fetch_cves_skips_missing_cvss(sample_cves):
    """CVEs without cvssMetricV31 are silently skipped."""
    no_cvss_cve = [s for s in sample_cves if s["cve"]["id"] == "CVE-2024-45678"]
    assert no_cvss_cve, "sample data must include CVE-2024-45678"

    payload = _make_nvd_response(no_cvss_cve)
    client = NVDClient(cache_path=None)

    with patch.object(client, "_fetch_page", return_value=payload):
        results = client.fetch_cves(max_results=5)

    assert results == []


# ---------------------------------------------------------------------------
# test_fetch_cves_detects_exploit_references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "references,expected",
    [
        (["https://github.com/user/repo-poc"], True),
        (["https://www.exploit-db.com/exploits/12345"], True),
        (["https://example.com/poc-demo"], True),
        (["https://nvd.nist.gov/vuln/detail/CVE-2024-0001"], False),
        ([], False),
        (
            [
                "https://vendor.example.com/advisory",
                "https://github.com/researcher/CVE-2024-poc",
            ],
            True,
        ),
    ],
)
def test_fetch_cves_detects_exploit_references(sample_cves, references, expected):
    """has_exploit_ref is True when any reference URL contains 'exploit', 'poc', or 'github.com'."""
    cve_entry = json.loads(json.dumps(sample_cves[0]))  # deep copy
    cve_entry["cve"]["references"] = [{"url": u} for u in references]
    payload = _make_nvd_response([cve_entry])
    client = NVDClient(cache_path=None)

    with patch.object(client, "_fetch_page", return_value=payload):
        results = client.fetch_cves(max_results=5)

    assert len(results) == 1
    assert results[0]["has_exploit_ref"] is expected


# ---------------------------------------------------------------------------
# test_fetch_cves_retries_on_failure
# ---------------------------------------------------------------------------


def test_fetch_cves_retries_on_failure(sample_cves):
    """_fetch_page retries 3 times on HTTP errors, then returns None."""
    client = NVDClient(cache_path=None)

    with (
        patch("time.sleep"),  # suppress real sleeps
        patch.object(client, "_api_reachable", return_value=True),
        patch("httpx.Client") as mock_client_cls,
    ):
        mock_http = MagicMock()
        mock_http.__enter__ = lambda s: mock_http
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.get.side_effect = [
            _mock_httpx_response({}, status_code=503),
            _mock_httpx_response({}, status_code=503),
            _mock_httpx_response({}, status_code=503),
        ]
        mock_client_cls.return_value = mock_http

        results = client.fetch_cves(max_results=5)

    assert results == []
    assert mock_http.get.call_count == 3


def test_fetch_cves_succeeds_after_transient_failure(sample_cves):
    """_fetch_page succeeds on the third attempt after two transient failures."""
    payload = _make_nvd_response(sample_cves[:1])
    client = NVDClient(cache_path=None)

    with (
        patch("time.sleep"),
        patch.object(client, "_api_reachable", return_value=True),
        patch("httpx.Client") as mock_client_cls,
    ):
        mock_http = MagicMock()
        mock_http.__enter__ = lambda s: mock_http
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.get.side_effect = [
            _mock_httpx_response({}, status_code=503),
            _mock_httpx_response({}, status_code=429),
            _mock_httpx_response(payload, status_code=200),
        ]
        mock_client_cls.return_value = mock_http

        results = client.fetch_cves(max_results=5)

    assert len(results) == 1
    assert results[0]["cve_id"] == "CVE-2024-21762"
    assert mock_http.get.call_count == 3


# ---------------------------------------------------------------------------
# NVDRateLimiter unit tests
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_requests_within_limit():
    """Rate limiter does not sleep when under the request cap."""
    limiter = NVDRateLimiter(max_requests=5, window=30)
    with patch("time.sleep") as mock_sleep:
        for _ in range(5):
            limiter.wait()
        mock_sleep.assert_not_called()


def test_rate_limiter_sleeps_when_limit_exceeded():
    """Rate limiter sleeps when the request cap is reached within the window."""
    limiter = NVDRateLimiter(max_requests=2, window=30)
    with patch("time.sleep") as mock_sleep:
        with patch("time.monotonic", side_effect=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]):
            limiter.wait()  # request 1
            limiter.wait()  # request 2
            limiter.wait()  # request 3 — should trigger sleep
        mock_sleep.assert_called_once()
