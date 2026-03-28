---
globs: src/api/**/*.py
---

When working on API files:
- All endpoints use Pydantic schemas from schemas.py
- Return proper HTTP status codes (400 bad input, 404 not found, 500 internal)
- Structured error responses with error_type and detail fields
- Add docstrings compatible with FastAPI auto-docs
- API tests use FastAPI TestClient with mocked dependencies