# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tooling
RUN pip install --upgrade pip

# Copy dependency spec and install into a prefix so we can copy cleanly
COPY pyproject.toml .
COPY src/ ./src/

RUN pip install --prefix=/install --no-cache-dir ".[dev]" 2>/dev/null || \
    pip install --prefix=/install --no-cache-dir .

# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/knowledge_base/ ./data/knowledge_base/

# Create writable directories for runtime artifacts
RUN mkdir -p data/models data/processed data/metrics data/raw

# Non-root user for security
RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
