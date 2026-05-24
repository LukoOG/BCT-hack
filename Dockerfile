# ── BCT Hack — Dockerfile ─────────────────────────────────────────────────────
#
# Single image used by both services in docker-compose.yml.
# CMD is overridden per-service in docker-compose.
#
# Prerequisites (run ONCE on the host before `docker compose up`):
#   python scripts/build_all.py --skip-fetch
#
# Data is volume-mounted at runtime — never baked into the image.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    EMBEDDING_BACKEND=tfidf

# ── System deps ────────────────────────────────────────────────────────────────
# build-essential: C extensions (faiss-cpu, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ────────────────────────────────────────────────────────────────
COPY requirements-demilade.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-demilade.txt

# ── Application source ─────────────────────────────────────────────────────────
COPY app ./app
COPY scripts ./scripts
COPY deployment_data ./deployment_data
COPY main.py .
COPY startup.py .
COPY .env.example .

# ── Eval/EDA outputs (optional — for demo without a full rebuild) ──────────────
RUN mkdir -p notebooks/outputs
COPY notebooks/outputs ./notebooks/outputs

# ── Ports ──────────────────────────────────────────────────────────────────────
EXPOSE 8000 8501

# ── Healthcheck ────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Default command: API ───────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "app.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000"]