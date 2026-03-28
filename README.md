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
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure secrets
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Build knowledge base index
python -c "from rag.knowledge_base import KnowledgeBase; KnowledgeBase().build_index()"

# 5. Start the API
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

> **Train the model first:** `/predict` and `/analyze` require a trained model saved to `data/models/`. Run your training pipeline (see `src/models/severity_predictor.py`) before using those endpoints.

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "index_loaded": true, "version": "0.1.0"}
```

### Full Analysis (ML prediction + RAG playbook)
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"cve_id": "CVE-2024-21762"}'
```

### ML Prediction Only
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"cve_id": "CVE-2024-21762"}'
```
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

### RAG Playbook Generation
```bash
curl -X POST http://localhost:8000/playbook \
  -H "Content-Type: application/json" \
  -d '{
    "cve_id": "CVE-2024-21762",
    "description": "Out-of-bounds write in FortiOS allows unauthenticated RCE",
    "severity": "CRITICAL",
    "cwe": "CWE-787"
  }'
```

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
