FROM python:3.11-slim AS base

# System libraries needed by fastembed (OpenMP) and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM deps AS runtime

# Create non-root user for security.
RUN groupadd -r appuser && useradd -r -g appuser appuser && chown appuser:appuser /app && mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser sql/ ./sql/
COPY --chown=appuser:appuser etl/ ./etl/
COPY --chown=appuser:appuser telegram_bot.py .

# Pre-download the fastembed ONNX router model as root so the HF xet client
# can write its logs. Cache lands in /app/.fastembed_cache, then we chown it
# to appuser. HF scratch goes to /tmp and is discarded after the build step.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache
RUN HF_HOME=/tmp/hf_build python -c \
        "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5').embed(['warmup']))" \
    && chown -R appuser:appuser /app/.fastembed_cache \
    && rm -rf /tmp/hf_build

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["python", "telegram_bot.py"]
