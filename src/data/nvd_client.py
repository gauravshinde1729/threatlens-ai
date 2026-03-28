"""NIST NVD API 2.0 client for fetching and parsing CVE data."""

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
RESULTS_PER_PAGE = 100
MAX_REQUESTS = 5
RATE_WINDOW_SECONDS = 30
MAX_RETRIES = 3
PAGE_SLEEP_SECONDS = 6

_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "cves_cache.json"
_DEFAULT_RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "scripts" / "raw_data.json"
_API_CONNECT_TIMEOUT = 2.0  # seconds — fall back to file if server unreachable


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

    def __init__(
        self,
        timeout: float = 30.0,
        cache_path: Path | str | None = _DEFAULT_CACHE_PATH,
        raw_data_path: Path | str | None = _DEFAULT_RAW_DATA_PATH,
    ) -> None:
        self._timeout = timeout
        self._cache_path = Path(cache_path) if cache_path is not None else None
        self._raw_data_path = Path(raw_data_path) if raw_data_path is not None else None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_cves(self, max_results: int = 500) -> list[dict[str, Any]]:
        """Fetch CVEs with CVSS v3.1 data, using cache → API → raw file fallback.

        Priority order:
        1. Load from ``cache_path`` if it exists (skip network entirely).
        2. Probe the NVD API with a 2-second connect timeout.  If it responds,
           paginate and collect up to *max_results* CVEs with CVSS v3.1 data.
        3. If the API does not respond within 2 seconds, parse ``raw_data.json``
           from the scripts/ directory instead.

        Results from step 2 or 3 are written to the cache for future runs.

        Args:
            max_results: Hard cap on CVEs returned (only those with CVSS v3.1).

        Returns:
            List of parsed CVE dicts.
        """
        # 1. Cache hit
        if self._cache_path is not None and self._cache_path.exists():
            logger.info("Loading CVEs from cache: %s", self._cache_path)
            cached: list[dict[str, Any]] = json.loads(
                self._cache_path.read_text(encoding="utf-8")
            )
            logger.info("Loaded %d CVEs from cache", len(cached))
            return cached

        # 2. API (2-second probe first — fall back immediately if unreachable)
        if self._api_reachable():
            results = self._paginate_api(max_results)
        else:
            logger.warning(
                "NVD API did not respond within %.0fs — loading from %s",
                _API_CONNECT_TIMEOUT,
                self._raw_data_path,
            )
            results = self._load_from_raw_file(max_results)

        # 3. Save whatever we got to cache for next run
        if results and self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            logger.info("Cached %d CVEs → %s", len(results), self._cache_path)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _api_reachable(self) -> bool:
        """Return True if the NVD API responds within *_API_CONNECT_TIMEOUT* seconds."""
        try:
            with httpx.Client(timeout=_API_CONNECT_TIMEOUT) as c:
                c.get(NVD_API_URL, params={"resultsPerPage": 1, "startIndex": 0})
            return True
        except Exception:
            return False

    def _paginate_api(self, max_results: int) -> list[dict[str, Any]]:
        """Paginate the NVD API and return CVEs with CVSS v3.1 data."""
        logger.info("Fetching up to %d CVEs (CVSS v3.1 only) from NVD API", max_results)
        results: list[dict[str, Any]] = []
        start_index = 0
        first_page = True

        with httpx.Client(timeout=self._timeout) as client:
            while len(results) < max_results:
                if not first_page:
                    time.sleep(PAGE_SLEEP_SECONDS)
                first_page = False

                page = self._fetch_page(client, start_index)
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
                    cve_obj = item.get("cve", {})
                    if not cve_obj.get("metrics", {}).get("cvssMetricV31"):
                        continue
                    parsed = self._parse_cve(cve_obj)
                    if parsed:
                        results.append(parsed)

                start_index += RESULTS_PER_PAGE
                if start_index >= total_results or not vulnerabilities:
                    break

        logger.info("Fetched %d CVEs with CVSS v3.1 from NVD API", len(results))
        return results

    def _load_from_raw_file(self, max_results: int) -> list[dict[str, Any]]:
        """Parse CVEs from ``raw_data.json``, keeping only CVSS v3.1 entries."""
        if self._raw_data_path is None or not self._raw_data_path.exists():
            logger.error("raw_data_path not set or file missing: %s", self._raw_data_path)
            return []

        logger.info("Parsing CVEs from raw data file: %s", self._raw_data_path)
        feed = json.loads(self._raw_data_path.read_text(encoding="utf-8"))
        vulnerabilities = feed.get("vulnerabilities", [])
        logger.info("Raw file contains %d total entries", len(vulnerabilities))

        results: list[dict[str, Any]] = []
        for item in vulnerabilities:
            if len(results) >= max_results:
                break
            cve_obj = item.get("cve", {})
            if not cve_obj.get("metrics", {}).get("cvssMetricV31"):
                continue
            parsed = self._parse_cve(cve_obj)
            if parsed:
                results.append(parsed)

        logger.info("Loaded %d CVEs with CVSS v3.1 from raw file", len(results))
        return results

    def _fetch_page(
        self,
        client: httpx.Client,
        start_index: int,
    ) -> dict[str, Any] | None:
        """Fetch a single page from the NVD API with retry + backoff."""
        params = {
            "resultsPerPage": RESULTS_PER_PAGE,
            "startIndex": start_index,
        }

        for attempt in range(1, MAX_RETRIES + 1):
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
