# ThreatLens AI

> **Predict CVE exploitability with ML, generate remediation playbooks with RAG**

ThreatLens AI is an end-to-end threat intelligence platform that ingests CVEs from NIST NVD, predicts whether they will be exploited in the wild using an ML ensemble, and generates actionable remediation playbooks using a Retrieval-Augmented Generation pipeline.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ThreatLens AI                               │
│                                                                       │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │  NVD API │───▶│ CVE Ingestion│───▶│   Feature Preprocessor   │   │
│  │ (NIST)   │    │ (nvd_client) │    │  ordinal + keyword + CWE  │   │
│  └──────────┘    └──────────────┘    └─────────────┬────────────┘   │
│                                                     │                 │
│                         ┌───────────────────────────▼─────────────┐  │
│                         │       ML Ensemble (Soft Voting)          │  │
│                         │  RandomForest + XGBoost → SHAP explain   │  │
│                         └───────────────────────────┬─────────────┘  │
│                                                     │                 │
│  ┌──────────────────────┐            ┌──────────────▼─────────────┐  │
│  │   Knowledge Base     │            │       RAG Pipeline          │  │
│  │  (10 security docs)  │──FAISS────▶│  SecurityRetriever +        │  │
│  │  sentence-transformers│           │  PlaybookGenerator (Groq)   │  │
│  └──────────────────────┘           └──────────────┬─────────────┘  │
│                                                     │                 │
│                         ┌───────────────────────────▼─────────────┐  │
│                         │          FastAPI REST API                 │  │
│                         │   POST /analyze   /predict   /playbook   │  │
│                         └───────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/your-org/threatlens-ai.git && cd threatlens-ai
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure secrets
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Train the model + build the FAISS index (one-time setup, ~5 mins)
python scripts/train_pipeline.py

# 5. Start the API
python -m uvicorn src.api.main:app --port 8002
```

API docs available at `http://localhost:8002/docs`

> **Note:** The server takes ~15 seconds to start — wait for `Application startup complete.` before sending requests.

---

## API Usage

All examples use port `8002`. Replace with your actual port if different.

> **Windows users:** Run these in **PowerShell** (not CMD). The single-line format works on Windows, macOS, and Linux.

---

### 1. Health Check
```bash
curl http://localhost:8002/health
```
Expected response:
```json
{"status":"ok","model_loaded":true,"index_loaded":true,"version":"0.1.0"}
```

---

### 2. POST /predict — ML exploit probability

Fastest option — pass features directly, no network lookup required:
```bash
curl -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d "{\"cve_id\": \"CVE-2024-21762\", \"features\": {\"cvss_v3_score\": 9.8, \"attack_vector\": 3, \"attack_complexity\": 1, \"privileges_required\": 2, \"user_interaction\": 1, \"scope\": 0, \"confidentiality_impact\": 2, \"integrity_impact\": 2, \"availability_impact\": 2, \"description_length\": 180, \"reference_count\": 3, \"affected_product_count\": 2, \"days_since_publication\": 400, \"has_exploit_ref\": 1, \"has_keyword_rce\": 1, \"has_keyword_sqli\": 0, \"has_keyword_xss\": 0, \"has_keyword_auth_bypass\": 0, \"has_keyword_buffer_overflow\": 1, \"has_keyword_privilege_escalation\": 0, \"cwe_79\": 0, \"cwe_89\": 0, \"cwe_787\": 1, \"cwe_416\": 0, \"cwe_78\": 0, \"cwe_20\": 0, \"cwe_125\": 0, \"cwe_476\": 0, \"cwe_190\": 0, \"cwe_119\": 0, \"cwe_other\": 0}}"
```
Expected response:
```json
{
  "cve_id": "CVE-2024-21762",
  "cvss_score": 9.8,
  "exploit_probability": 0.94,
  "risk_level": "HIGH",
  "shap_explanation": {
    "top_positive_features": [["has_exploit_ref", 0.31], ["cvss_v3_score", 0.18]],
    "top_negative_features": [["user_interaction", -0.12]]
  }
}
```

Alternative — look up by CVE ID from local cache (no features needed):
```bash
curl -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d "{\"cve_id\": \"CVE-2024-21762\"}"
```
> Returns 404 if the CVE is not in `data/raw/cves_cache.json`.

---

### 3. POST /playbook — RAG remediation playbook

```bash
curl -X POST http://localhost:8002/playbook -H "Content-Type: application/json" -d "{\"cve_id\": \"CVE-2024-21762\", \"description\": \"Out-of-bounds write in FortiOS allows unauthenticated remote code execution via crafted HTTP requests\", \"severity\": \"CRITICAL\", \"cwe\": \"CWE-787\"}"
```
> Requires `GROQ_API_KEY` set in `.env`. Returns 503 without it.

---

### 4. POST /analyze — Full pipeline (predict + playbook)

```bash
curl -X POST http://localhost:8002/analyze -H "Content-Type: application/json" -d "{\"cve_id\": \"CVE-2024-21762\"}"
```
> Looks up the CVE from local cache then runs ML prediction + RAG playbook generation. Returns 404 if the CVE ID is not in `data/raw/cves_cache.json`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Data ingestion | NIST NVD API 2.0, `httpx` (rate-limited, retry) |
| Feature engineering | `pandas`, ordinal + keyword + CWE one-hot |
| ML ensemble | `scikit-learn` RandomForest + `XGBoost`, soft voting |
| Explainability | `shap` TreeExplainer |
| Embeddings | `sentence-transformers` all-MiniLM-L6-v2 |
| Vector store | `faiss-cpu` IndexFlatL2 |
| RAG orchestration | `langchain`, `langchain-groq` |
| LLM inference | Groq API — llama-3.3-70b-versatile (free tier) |
| REST API | `FastAPI` + `pydantic` v2 + `uvicorn` |
| Monitoring | PSI drift detection, JSON prediction log |
| Containerisation | Docker multi-stage + Docker Compose |
| Testing | `pytest` + `pytest-cov` (≥80% coverage) |
| Linting | `ruff` |

---

## Project Structure

```
threatlens-ai/
├── configs/
│   └── model_config.yaml          # All hyperparameters — no hardcoding
├── data/
│   ├── knowledge_base/            # 10 security remediation markdown docs
│   ├── models/                    # Versioned joblib model artifacts
│   └── processed/faiss_index/    # Built FAISS vector index
├── src/
│   ├── api/
│   │   ├── main.py                # FastAPI app, lifespan, CORS, error handlers
│   │   ├── schemas.py             # Pydantic request/response models
│   │   ├── dependencies.py        # Shared app state singleton
│   │   └── routes/                # analyze.py, predict.py, playbook.py
│   ├── data/
│   │   ├── nvd_client.py          # NVD API 2.0 client (pagination + rate limit)
│   │   ├── preprocessor.py        # Feature engineering pipeline
│   │   └── feature_store.py       # CSV feature persistence
│   ├── models/
│   │   ├── severity_predictor.py  # RF + XGBoost ensemble + SHAP
│   │   ├── cve_clusterer.py       # DBSCAN campaign grouping
│   │   ├── text_classifier.py     # sentence-transformers classifier
│   │   └── model_registry.py      # Versioned joblib persistence
│   ├── rag/
│   │   ├── knowledge_base.py      # FAISS index builder/loader
│   │   ├── retriever.py           # CVE-aware semantic retrieval
│   │   ├── playbook_generator.py  # LLM playbook generation
│   │   └── prompts/               # Prompt templates
│   ├── evaluation/
│   │   ├── metrics.py             # Full sklearn evaluation suite
│   │   └── explainability.py      # SHAP plot + top feature extraction
│   └── monitoring/
│       ├── drift_detector.py      # PSI-based feature drift detection
│       └── performance_tracker.py # Prediction latency/error JSON log
├── tests/                         # 100+ tests, all external APIs mocked
└── docker/
    ├── Dockerfile                 # Multi-stage python:3.11-slim
    └── docker-compose.yml
```

---

## Design Decisions

- **Why predict exploitability beyond CVSS?** CVSS scores measure theoretical severity, not real-world exploit likelihood. Our ensemble trains on exploit references, CWE patterns, and attack characteristics — features that correlate with actual in-the-wild exploitation (CISA KEV data). A CVSS 9.8 CVE requiring physical access is far less urgent than a CVSS 7.5 network-exploitable one with a public PoC.

- **Why RAG over fine-tuning?** Fine-tuning an LLM on security playbooks requires thousands of labeled examples, GPU compute, and re-training whenever guidance changes. RAG lets us update the knowledge base (add a doc, edit a procedure) without touching the model. It also keeps the LLM grounded — retrieved context anchors output in vetted internal documentation rather than hallucinated best practices.

- **Why RF + XGBoost ensemble over a single model?** Random Forest provides stable, low-variance predictions with native feature importance via SHAP; XGBoost captures non-linear interactions and typically achieves higher raw accuracy. Soft-voting their probability outputs combines both strengths and reduces the chance that either model's blind spots dominate. In CVE scoring, false negatives (missing a weaponized vulnerability) are costlier than false positives.

---

## Development

```bash
make test      # pytest with coverage (min 80%)
make lint      # ruff check
make format    # ruff format
```

---

## License

MIT
