"""FAISS-backed knowledge base built from security remediation markdown files."""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "faiss_index"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_EMBED_DIM = 384
_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50


class KnowledgeBase:
    """Load markdown security docs, chunk them, embed with sentence-transformers,
    and store in a FAISS flat-L2 index for similarity retrieval."""

    def __init__(
        self,
        kb_dir: str | Path = _DEFAULT_KB_DIR,
        index_dir: str | Path = _DEFAULT_INDEX_DIR,
    ) -> None:
        self.kb_dir = Path(kb_dir)
        self.index_dir = Path(index_dir)
        self._encoder = SentenceTransformer(_EMBED_MODEL)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            length_function=len,
        )
        # Populated after build_index() or load_index()
        self._index: faiss.IndexFlatL2 | None = None
        self._chunks: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """Load all .md files, chunk, embed, and store in a FAISS index.

        Saves the index and chunk metadata to *index_dir* for later reuse.
        """
        md_files = sorted(self.kb_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No .md files found in {self.kb_dir}")

        logger.info("Building index from %d markdown files in %s", len(md_files), self.kb_dir)
        self._chunks = []
        self._metadata = []

        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            raw_chunks = self._splitter.split_text(text)
            for idx, chunk in enumerate(raw_chunks):
                self._chunks.append(chunk)
                self._metadata.append({"source_file": md_path.name, "chunk_index": idx})

        logger.info("Embedding %d chunks…", len(self._chunks))
        embeddings = self._encoder.encode(self._chunks, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        self._index = faiss.IndexFlatL2(_EMBED_DIM)
        self._index.add(embeddings)

        self._save_index()
        logger.info("Index built: %d vectors", self._index.ntotal)

    def load_index(self) -> None:
        """Load a previously built FAISS index and chunk metadata from disk."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "chunks.pkl"

        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self.index_dir}. Run build_index() first."
            )

        self._index = faiss.read_index(str(index_path))
        with meta_path.open("rb") as f:
            saved = pickle.load(f)
        self._chunks = saved["chunks"]
        self._metadata = saved["metadata"]
        logger.info("Loaded index: %d vectors, %d chunks", self._index.ntotal, len(self._chunks))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the *top_k* most similar chunks to *query*.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: content, source_file, chunk_index,
            similarity_score (lower L2 distance = more similar).
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        q_emb = self._encoder.encode([query], show_progress_bar=False).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx == -1:
                continue
            results.append(
                {
                    "content": self._chunks[idx],
                    "source_file": self._metadata[idx]["source_file"],
                    "chunk_index": self._metadata[idx]["chunk_index"],
                    "similarity_score": float(dist),
                }
            )
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the current index.

        Returns:
            Dict with doc_count, chunk_count, and index_size (vectors stored).
        """
        source_files = {m["source_file"] for m in self._metadata}
        return {
            "doc_count": len(source_files),
            "chunk_count": len(self._chunks),
            "index_size": self._index.ntotal if self._index is not None else 0,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        meta_path = self.index_dir / "chunks.pkl"
        with meta_path.open("wb") as f:
            pickle.dump({"chunks": self._chunks, "metadata": self._metadata}, f)
        logger.info("Index saved to %s", self.index_dir)
