"""Shared application state and FastAPI dependency helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.severity_predictor import SeverityPredictor
    from rag.knowledge_base import KnowledgeBase
    from rag.retriever import SecurityRetriever


@dataclass
class AppState:
    """Container for lazily-loaded singletons shared across route handlers."""

    predictor: SeverityPredictor | None = field(default=None)
    knowledge_base: KnowledgeBase | None = field(default=None)
    retriever: SecurityRetriever | None = field(default=None)
    model_loaded: bool = False
    index_loaded: bool = False


# Module-level singleton — populated during application startup
app_state = AppState()
