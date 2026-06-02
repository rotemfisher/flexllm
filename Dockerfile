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
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM deps AS runtime

# Create non-root user for security.
RUN groupadd -r appuser && useradd -r -g appuser appuser && chown appuser:appuser /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser sql/ ./sql/
COPY --chown=appuser:appuser etl/ ./etl/
COPY --chown=appuser:appuser telegram_bot.py .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["python", "telegram_bot.py"]
