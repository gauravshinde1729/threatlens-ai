---
globs: src/rag/**/*.py
---

When working on RAG pipeline files:
- Use LangChain for chain orchestration
- FAISS for vector index, sentence-transformers for embeddings
- LLM provider: Groq (free inference) via langchain-groq package
- Model: llama-3.3-70b-versatile (or mixtral-8x7b-32768 as fallback)
- Groq API key via GROQ_API_KEY environment variable
- Chunk size: 500 tokens with 50 token overlap
- Always include source metadata with retrieved documents
- Mock Groq API calls in all tests
- Prompt templates in src/rag/prompts/ as separate Python files