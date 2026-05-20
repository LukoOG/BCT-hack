FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-demilade.txt .
RUN pip install --no-cache-dir -r requirements-demilade.txt

COPY app ./app
COPY scripts ./scripts
COPY main.py .
COPY .env.example .

# Optional baked-in eval/EDA outputs for demo without a full rebuild
COPY notebooks/outputs ./notebooks/outputs

EXPOSE 8000 8501

# Default: API. Override in docker-compose for Streamlit.
CMD ["python", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
