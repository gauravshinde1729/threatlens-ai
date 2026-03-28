# ThreatLens AI — Project Conventions

## What this project is
End-to-end threat intelligence platform: ingest CVEs from NIST NVD,
predict exploit likelihood with ML, generate remediation playbooks with RAG + LLM.

## Language & tools
- Python 3.14, managed with pyproject.toml
- Virtual env in .venv/
- Formatter: ruff format
- Linter: ruff check
- Tests: pytest with coverage (minimum 80%)
- Type hints required on all public functions

## Architecture
- src/ layout with clear module boundaries
- FastAPI for REST API
- Pydantic for all data validation
- YAML configs in configs/ — no hardcoded hyperparameters
- AWS services mocked locally (S3, SageMaker, Bedrock)

## ML conventions
- All models implement fit(), predict(), and explain() methods
- Use scikit-learn Pipeline where possible
- Save models with joblib via model_registry
- SHAP TreeExplainer for tree-based models
- XGBoost + RandomForest ensemble for severity prediction

## RAG conventions
- LangChain for orchestration
- FAISS for vector storage
- sentence-transformers for embeddings
- Groq for LLM inference (free tier, llama-3.3-70b-versatile)
- GROQ_API_KEY environment variable for authentication
- Knowledge base docs in data/knowledge_base/

## Testing
- pytest fixtures for sample CVE data
- Mock external APIs (NVD, Claude) in tests — never call real services
- Tests mirror src/ structure in tests/ directory

## Git
- Conventional commits: feat:, fix:, docs:, test:, ci:
- PR branches from main, squash merge