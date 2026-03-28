"""Tests for the RAG pipeline: KnowledgeBase, SecurityRetriever, PlaybookGenerator."""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.knowledge_base import KnowledgeBase
from rag.retriever import SecurityRetriever

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kb_dir(tmp_path) -> Path:
    """Minimal knowledge base directory with 3 markdown files."""
    docs = {
        "sql_injection.md": textwrap.dedent("""\
            # SQL Injection Remediation
            Use parameterized queries to prevent SQL injection.
            Never concatenate user input into SQL strings.
            Prepared statements are the primary defense against CWE-89.
            Database hardening includes least-privilege accounts.
        """),
        "command_injection.md": textwrap.dedent("""\
            # Command Injection Guide (CWE-78)
            Avoid shell=True in subprocess calls.
            Use argument arrays instead of string interpolation.
            Input validation and allowlisting are critical defenses.
            WAF rules block semicolons and pipe characters.
        """),
        "xss_prevention.md": textwrap.dedent("""\
            # XSS Prevention Guide (CWE-79)
            Output encoding prevents cross-site scripting attacks.
            Content Security Policy headers restrict script execution.
            Use DOMPurify for sanitizing user-generated HTML content.
            HttpOnly cookie flags protect session tokens from theft.
        """),
    }
    for name, content in docs.items():
        (tmp_path / name).write_text(content)
    return tmp_path


@pytest.fixture()
def built_kb(kb_dir, tmp_path) -> KnowledgeBase:
    """KnowledgeBase with index built from the tmp markdown files."""
    index_dir = tmp_path / "faiss_index"
    kb = KnowledgeBase(kb_dir=kb_dir, index_dir=index_dir)
    kb.build_index()
    return kb


@pytest.fixture()
def retriever(built_kb) -> SecurityRetriever:
    return SecurityRetriever(built_kb)


# ---------------------------------------------------------------------------
# test_knowledge_base_builds_index_from_markdown_files
# ---------------------------------------------------------------------------


def test_knowledge_base_builds_index_from_markdown_files(built_kb):
    """build_index() loads all .md files and stores embeddings in FAISS."""
    stats = built_kb.get_stats()
    assert stats["doc_count"] == 3
    assert stats["chunk_count"] > 0
    assert stats["index_size"] == stats["chunk_count"]


def test_knowledge_base_stats_structure(built_kb):
    """get_stats() returns the required keys."""
    stats = built_kb.get_stats()
    assert "doc_count" in stats
    assert "chunk_count" in stats
    assert "index_size" in stats


def test_knowledge_base_raises_on_empty_directory(tmp_path):
    """build_index() raises FileNotFoundError when no .md files are present."""
    kb = KnowledgeBase(kb_dir=tmp_path, index_dir=tmp_path / "idx")
    with pytest.raises(FileNotFoundError):
        kb.build_index()


# ---------------------------------------------------------------------------
# test_retriever_returns_relevant_chunks_with_metadata
# ---------------------------------------------------------------------------


def test_retriever_returns_relevant_chunks_with_metadata(retriever):
    """retrieve() returns a list of dicts with all required metadata fields."""
    results = retriever.retrieve("sql injection parameterized queries", top_k=3)

    assert len(results) <= 3
    assert len(results) > 0

    for result in results:
        assert "content" in result
        assert "source_file" in result
        assert "chunk_index" in result
        assert "similarity_score" in result

        assert isinstance(result["content"], str)
        assert isinstance(result["source_file"], str)
        assert isinstance(result["chunk_index"], int)
        assert isinstance(result["similarity_score"], float)


def test_retriever_returns_relevant_source_for_sql_query(retriever):
    """SQL injection query should surface the sql_injection.md document."""
    results = retriever.retrieve("SQL injection parameterized queries CWE-89", top_k=5)
    sources = [r["source_file"] for r in results]
    assert any("sql" in src.lower() for src in sources), (
        f"Expected sql_injection.md in results, got: {sources}"
    )


def test_retriever_returns_relevant_source_for_xss_query(retriever):
    """XSS query should surface the xss_prevention.md document."""
    results = retriever.retrieve("cross-site scripting output encoding CSP", top_k=5)
    sources = [r["source_file"] for r in results]
    assert any("xss" in src.lower() for src in sources), (
        f"Expected xss_prevention.md in results, got: {sources}"
    )


def test_retriever_respects_top_k(retriever):
    """retrieve() never returns more than top_k results."""
    for k in (1, 2, 3):
        results = retriever.retrieve("injection vulnerability", top_k=k)
        assert len(results) <= k


# ---------------------------------------------------------------------------
# test_retriever_for_cve_builds_smart_query
# ---------------------------------------------------------------------------


def test_retriever_for_cve_builds_smart_query(retriever):
    """retrieve_for_cve() returns results and logs a CWE-enriched query."""
    cve = {
        "cve_id": "CVE-2024-12345",
        "cwe_ids": ["CWE-89"],
        "description": "SQL injection in login endpoint allows auth bypass.",
        "attack_vector": "NETWORK",
    }
    results = retriever.retrieve_for_cve(cve, top_k=3)

    assert len(results) > 0
    # SQL injection CWE hint should surface the SQL doc
    sources = [r["source_file"] for r in results]
    assert any("sql" in src.lower() for src in sources)


def test_retriever_for_cve_handles_missing_cwe(retriever):
    """retrieve_for_cve() does not crash when cwe_ids is empty."""
    cve = {
        "cve_id": "CVE-2024-99999",
        "cwe_ids": [],
        "description": "An unclassified vulnerability in a web application.",
        "attack_vector": "NETWORK",
    }
    results = retriever.retrieve_for_cve(cve, top_k=3)
    assert isinstance(results, list)


def test_retriever_for_cve_command_injection_cwe(retriever):
    """CWE-78 maps to command injection hint, surfacing command_injection.md."""
    cve = {
        "cve_id": "CVE-2024-CWE78",
        "cwe_ids": ["CWE-78"],
        "description": "OS command injection via unsanitized filename parameter.",
        "attack_vector": "NETWORK",
    }
    results = retriever.retrieve_for_cve(cve, top_k=5)
    sources = [r["source_file"] for r in results]
    assert any("command" in src.lower() for src in sources)


# ---------------------------------------------------------------------------
# test_playbook_generator_produces_all_sections (mocked Groq)
# ---------------------------------------------------------------------------


def test_playbook_generator_produces_all_sections(built_kb, tmp_path):
    """PlaybookGenerator.generate() formats prompt correctly and returns playbook dict."""
    from rag.playbook_generator import PlaybookGenerator

    mock_playbook_text = """
## 1. Executive Summary
This critical vulnerability allows remote code execution without authentication.

## 2. Risk Assessment
CVSS 9.8 — critical severity, network-exploitable, no privileges required.

## 3. Immediate Actions
1. Apply vendor patch immediately.
2. Enable WAF rules blocking exploit patterns.
3. Rotate credentials on affected systems.

## 4. Detection Rules
Monitor for: unusual child processes from web server, exploit signatures in WAF logs.

## 5. Monitoring Recommendations
Enable enhanced logging for 30 days post-patch.
""".strip()

    mock_response = MagicMock()
    mock_response.content = mock_playbook_text

    retriever = SecurityRetriever(built_kb)

    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-123"}),
        patch("rag.playbook_generator.ChatGroq") as mock_groq_cls,
    ):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_groq_cls.return_value = mock_llm

        generator = PlaybookGenerator(retriever)
        cve = {
            "cve_id": "CVE-2024-21762",
            "description": "Remote code execution in FortiOS via crafted HTTP request.",
            "cvss_v3_score": 9.8,
            "attack_vector": "NETWORK",
            "cwe_ids": ["CWE-787"],
            "affected_products": ["cpe:2.3:o:fortinet:fortios:7.4.0"],
            "has_exploit_ref": True,
            "published_date": "2024-02-09T00:00:00.000",
        }
        ml_prediction = {
            "exploit_probability": 0.92,
            "predicted_label": 1,
            "confidence": "high",
        }
        docs = retriever.retrieve_for_cve(cve, top_k=3)
        result = generator.generate(cve, ml_prediction, docs)

    assert result["cve_id"] == "CVE-2024-21762"
    assert isinstance(result["playbook"], str)
    assert len(result["playbook"]) > 0
    assert isinstance(result["sources"], list)
    assert result["model"] is not None

    playbook = result["playbook"]
    for section in ("Executive Summary", "Risk Assessment", "Immediate Actions",
                    "Detection Rules", "Monitoring Recommendations"):
        assert section in playbook, f"Missing section: {section}"


def test_playbook_generator_raises_without_api_key():
    """PlaybookGenerator raises EnvironmentError when GROQ_API_KEY is not set."""
    from rag.playbook_generator import PlaybookGenerator

    mock_kb = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever._kb = mock_kb

    with patch.dict("os.environ", {}, clear=True):
        # Remove GROQ_API_KEY if present
        import os
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            PlaybookGenerator(mock_retriever)


# ---------------------------------------------------------------------------
# test_knowledge_base_save_and_load_index_roundtrip
# ---------------------------------------------------------------------------


def test_knowledge_base_save_and_load_index_roundtrip(kb_dir, tmp_path):
    """build_index() + load_index() produces identical search results."""
    index_dir = tmp_path / "faiss_index"
    kb_build = KnowledgeBase(kb_dir=kb_dir, index_dir=index_dir)
    kb_build.build_index()
    results_before = kb_build.search("sql injection parameterized", top_k=3)

    kb_load = KnowledgeBase(kb_dir=kb_dir, index_dir=index_dir)
    kb_load.load_index()
    results_after = kb_load.search("sql injection parameterized", top_k=3)

    assert len(results_before) == len(results_after)
    for r_before, r_after in zip(results_before, results_after, strict=True):
        assert r_before["source_file"] == r_after["source_file"]
        assert r_before["chunk_index"] == r_after["chunk_index"]
        assert abs(r_before["similarity_score"] - r_after["similarity_score"]) < 1e-4


def test_load_index_raises_when_not_built(tmp_path):
    """load_index() raises FileNotFoundError when index files don't exist."""
    kb = KnowledgeBase(kb_dir=tmp_path, index_dir=tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        kb.load_index()


def test_search_raises_before_index_built(kb_dir, tmp_path):
    """search() raises RuntimeError if called before build_index/load_index."""
    kb = KnowledgeBase(kb_dir=kb_dir, index_dir=tmp_path / "idx")
    with pytest.raises(RuntimeError, match="Index not built"):
        kb.search("test query")
