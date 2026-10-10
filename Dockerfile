# Production image for the NexusNode web app.
# Secrets are never baked in: pass RIOT_KEY at runtime (e.g. the host's
# secrets manager). The image contains only the app, engine and model files.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-app.txt .
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-app.txt

# Only what the app needs at runtime (see .dockerignore for exclusions)
COPY app.py .
COPY modules/ modules/
COPY data/processed/nexus_model.pt data/processed/model_metrics.json \
     data/processed/champion_roles.json data/processed/champion_matchups.csv data/processed/
COPY data/static/ data/static/

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 nexus && chown -R nexus:nexus /app
USER nexus

ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8501\")}/_stcore/health', timeout=4)"

# Production settings: no tracebacks shown to visitors, XSRF protection on,
# no Streamlit telemetry, no file uploads accepted.
# exec: Streamlit becomes PID 1 and receives stop signals directly (clean shutdowns)
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --server.enableXsrfProtection=true --server.maxUploadSize=1 --client.showErrorDetails=none --client.toolbarMode=viewer --browser.gatherUsageStats=false"]
