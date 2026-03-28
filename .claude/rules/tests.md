---
globs: tests/**/*.py
---

When working on test files:
- Use pytest fixtures, never unittest.TestCase
- Mock external APIs (NVD, Claude) — never call real services
- Use sample CVE fixtures from tests/test_data/sample_cves.json
- Test naming: test_[method]_[scenario]_[expected_result]
- Each test function tests one behavior only
- Use parametrize for testing multiple inputs