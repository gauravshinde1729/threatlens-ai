# ThreatLens AI

End-to-end threat intelligence platform: ingest CVEs from NIST NVD, predict exploit likelihood with ML, and generate remediation playbooks with RAG + LLM.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        ThreatLens AI                            │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  NIST    │    │   Data   │    │    ML    │    │   RAG    │ │
│  │  NVD     │───▶│ Ingestion│───▶│  Models  │───▶│ Pipeline │ │
│  │  API     │    │          │    │          │    │          │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                       │               │               │        │
│                       ▼               ▼               ▼        │
│                  ┌─────────────────────────────────────────┐   │
│                  │              FastAPI REST API            │   │
│                  └─────────────────────────────────────────┘   │
│                                                                 │
│  ML Stack: XGBoost + RandomForest ensemble, SHAP explanations  │
│  RAG Stack: LangChain + FAISS + sentence-transformers + Groq   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Configure environment
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# 3. Run the API server
make run
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/cves/ingest` | Ingest a CVE by ID |
| `GET` | `/cves/{cve_id}` | Retrieve a CVE record |
| `POST` | `/predict` | Predict exploit likelihood for a CVE |
| `POST` | `/remediate` | Generate a remediation playbook |

> Full interactive documentation available at `/docs` when the server is running.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| ML models | XGBoost, scikit-learn (RandomForest ensemble) |
| Explainability | SHAP TreeExplainer |
| RAG orchestration | LangChain |
| Vector store | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM inference | Groq (`llama-3.3-70b-versatile`) |
| CVE data source | NIST NVD API |
| Linter / formatter | Ruff |
| Testing | pytest + pytest-cov |
| CI | GitHub Actions |

## Development

```bash
make test      # run pytest with coverage (min 80%)
make lint      # ruff check
make format    # ruff format
```

## License

MIT
