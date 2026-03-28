"""NIST NVD API 2.0 client for fetching and parsing CVE data."""

import logging
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 100
MAX_REQUESTS = 5
RATE_WINDOW_SECONDS = 30
MAX_RETRIES = 3


class NVDRateLimiter:
    """Sliding-window rate limiter: max 5 requests per 30 seconds."""

    def __init__(
        self, max_requests: int = MAX_REQUESTS, window: float = RATE_WINDOW_SECONDS
    ) -> None:
        self._max_requests = max_requests
        self._window = window
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        # Evict timestamps outside the window
        while self._timestamps and self._timestamps[0] <= now - self._window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_requests:
            sleep_for = self._window - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.debug("Rate limit reached, sleeping %.2fs", sleep_for)
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class NVDClient:
    """Client for the NIST NVD CVE API 2.0."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout
        self._rate_limiter = NVDRateLimiter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_cves(self, days_back: int = 90, max_results: int = 500) -> list[dict[str, Any]]:
        """Fetch CVEs modified within the last *days_back* days.

        Args:
            days_back: How many days back to query for last-modified CVEs.
            max_results: Hard cap on the total number of CVEs returned.

        Returns:
            List of parsed CVE dicts.
        """
        end_dt = datetime.now(tz=UTC)
        start_dt = end_dt - timedelta(days=days_back)
        start_iso = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000")
        end_iso = end_dt.strftime("%Y-%m-%dT%H:%M:%S.000")

        logger.info(
            "Fetching CVEs modified between %s and %s (max %d)",
            start_iso,
            end_iso,
            max_results,
        )

        results: list[dict[str, Any]] = []
        start_index = 0

        with httpx.Client(timeout=self._timeout) as client:
            while len(results) < max_results:
                page = self._fetch_page(client, start_iso, end_iso, start_index)
                if page is None:
                    break

                vulnerabilities = page.get("vulnerabilities", [])
                total_results = page.get("totalResults", 0)
                logger.debug(
                    "Page start_index=%d: got %d items (total=%d)",
                    start_index,
                    len(vulnerabilities),
                    total_results,
                )

                for item in vulnerabilities:
                    if len(results) >= max_results:
                        break
                    parsed = self._parse_cve(item.get("cve", {}))
                    if parsed:
                        results.append(parsed)

                start_index += RESULTS_PER_PAGE
                if start_index >= total_results or not vulnerabilities:
                    break

        logger.info("Fetched %d CVEs total", len(results))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(
        self,
        client: httpx.Client,
        start_iso: str,
        end_iso: str,
        start_index: int,
    ) -> dict[str, Any] | None:
        """Fetch a single page from the NVD API with retry + backoff."""
        params = {
            "lastModStartDate": start_iso,
            "lastModEndDate": end_iso,
            "resultsPerPage": RESULTS_PER_PAGE,
            "startIndex": start_index,
        }

        for attempt in range(1, MAX_RETRIES + 1):
            self._rate_limiter.wait()
            try:
                response = client.get(NVD_API_URL, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "HTTP %d on attempt %d/%d: %s",
                    exc.response.status_code,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
            except (httpx.RequestError, ValueError) as exc:
                logger.warning("Request error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)

            if attempt < MAX_RETRIES:
                backoff = 2 ** (attempt - 1)
                logger.debug("Backing off %ds before retry", backoff)
                time.sleep(backoff)

        logger.error("All %d attempts failed for start_index=%d", MAX_RETRIES, start_index)
        return None

    def _parse_cve(self, cve: dict[str, Any]) -> dict[str, Any] | None:
        """Parse a raw NVD CVE object into a clean dict."""
        cve_id = cve.get("id")
        if not cve_id:
            return None

        description = self._extract_description(cve)
        cvss = self._extract_cvss_v3(cve)
        references = self._extract_references(cve)
        cwe_ids = self._extract_cwe_ids(cve)
        affected_products = self._extract_affected_products(cve)

        return {
            "cve_id": cve_id,
            "description": description,
            "cvss_v3_score": cvss.get("baseScore"),
            "cvss_v3_vector": cvss.get("vectorString"),
            "attack_vector": cvss.get("attackVector"),
            "attack_complexity": cvss.get("attackComplexity"),
            "privileges_required": cvss.get("privilegesRequired"),
            "user_interaction": cvss.get("userInteraction"),
            "scope": cvss.get("scope"),
            "confidentiality_impact": cvss.get("confidentialityImpact"),
            "integrity_impact": cvss.get("integrityImpact"),
            "availability_impact": cvss.get("availabilityImpact"),
            "cwe_ids": cwe_ids,
            "references": references,
            "has_exploit_ref": self._has_exploit_ref(references),
            "published_date": cve.get("published"),
            "last_modified_date": cve.get("lastModified"),
            "affected_products": affected_products,
        }

    # ------------------------------------------------------------------
    # Field extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_description(cve: dict[str, Any]) -> str | None:
        descriptions = cve.get("descriptions", [])
        for d in descriptions:
            if d.get("lang") == "en":
                return d.get("value")
        return descriptions[0].get("value") if descriptions else None

    @staticmethod
    def _extract_cvss_v3(cve: dict[str, Any]) -> dict[str, Any]:
        """Return CVSS v3.x metrics dict; empty dict if not present."""
        metrics = cve.get("metrics", {})
        # Prefer v3.1 over v3.0
        for key in ("cvssMetricV31", "cvssMetricV30"):
            entries = metrics.get(key, [])
            if entries:
                return entries[0].get("cvssData", {})
        return {}

    @staticmethod
    def _extract_references(cve: dict[str, Any]) -> list[str]:
        return [ref.get("url", "") for ref in cve.get("references", []) if ref.get("url")]

    @staticmethod
    def _has_exploit_ref(references: list[str]) -> bool:
        keywords = ("exploit", "poc", "github.com")
        return any(any(kw in url.lower() for kw in keywords) for url in references)

    @staticmethod
    def _extract_cwe_ids(cve: dict[str, Any]) -> list[str]:
        cwe_ids: list[str] = []
        for weakness in cve.get("weaknesses", []):
            for desc in weakness.get("description", []):
                value = desc.get("value", "")
                if value and value != "NVD-CWE-Other" and value != "NVD-CWE-noinfo":
                    cwe_ids.append(value)
        return cwe_ids

    @staticmethod
    def _extract_affected_products(cve: dict[str, Any]) -> list[str]:
        products: list[str] = []
        configs = cve.get("configurations", [])
        for config in configs:
            for node in config.get("nodes", []):
                for match in node.get("cpeMatch", []):
                    if match.get("vulnerable"):
                        products.append(match.get("criteria", ""))
        return products
