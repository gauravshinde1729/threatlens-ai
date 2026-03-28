"""ThreatLens AI — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from api.dependencies import app_state  # noqa: E402
from api.routes import analyze, playbook, predict  # noqa: E402
from api.schemas import ErrorResponse, HealthResponse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and FAISS index on startup; clean up on shutdown."""
    _load_model()
    _load_index()
    yield
    logger.info("ThreatLens AI shutting down")


def _load_model() -> None:
    try:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry()
        versions = registry.list_versions("severity_predictor")
        if versions:
            latest = versions[-1]
            app_state.predictor = registry.load_model("severity_predictor", latest)
            app_state.model_loaded = True
            logger.info("Loaded severity_predictor v%s", latest)
        else:
            logger.warning("No trained model found — prediction endpoints will return 503")
    except Exception as exc:
        logger.warning("Could not load model: %s", exc)


def _load_index() -> None:
    try:
        from rag.knowledge_base import KnowledgeBase
        from rag.retriever import SecurityRetriever

        kb = KnowledgeBase()
        kb.load_index()
        app_state.knowledge_base = kb
        app_state.retriever = SecurityRetriever(kb)
        app_state.index_loaded = True
        logger.info("FAISS index loaded (%d vectors)", kb.get_stats()["index_size"])
    except FileNotFoundError:
        logger.warning("FAISS index not found — playbook endpoints will return 503")
    except Exception as exc:
        logger.warning("Could not load FAISS index: %s", exc)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


app = FastAPI(
    title="ThreatLens AI",
    description="Predict CVE exploitability with ML, generate remediation playbooks with RAG",
    version=_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, tags=["analysis"])
app.include_router(predict.router, tags=["prediction"])
app.include_router(playbook.router, tags=["playbook"])


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Return application health and readiness status."""
    return HealthResponse(
        status="ok",
        model_loaded=app_state.model_loaded,
        index_loaded=app_state.index_loaded,
        version=_VERSION,
    )


# ---------------------------------------------------------------------------
# Global exception handlers — structured JSON, no stack traces
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_type=type(exc).__name__,
            detail="An internal error occurred. Check server logs for details.",
        ).model_dump(),
    )
