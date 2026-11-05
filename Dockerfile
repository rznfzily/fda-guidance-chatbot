# Base image: slim Debian with Python 3.12 preinstalled.
FROM python:3.12-slim

# Python/pip settings: unbuffered logs, no .pyc files, no pip cache, no pip version check
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user owns /app
RUN useradd --create-home --uid 1000 appuser \
    && mkdir /app && chown appuser:appuser /app
# All following commands run from /app
WORKDIR /app

# Dependencies first for layer caching: this layer is reused unless requirements.txt changes
COPY requirements.txt ./
# CPU-only PyTorch first, pinned to a tested version (the default Linux wheel bundles CUDA and adds several GB);
# requirements.txt then sees it as satisfied. Long timeouts and retries: the wheel is about 210 MB.
ARG TORCH_VERSION=2.14.1
RUN pip install --default-timeout=100 --retries=5 "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --default-timeout=100 --retries=5 -r requirements.txt

# Reranker model baked into the image so the container needs no internet at runtime.
# RERANKER_MODEL at runtime must match this build arg, so build with the model named in your .env (see README.md);
# when deploying leave RERANKER_MODEL unset so the image's value is used.
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV RERANKER_MODEL=${RERANKER_MODEL} \
    HF_HOME=/opt/hf_cache
RUN python -c "import os; from sentence_transformers import CrossEncoder; CrossEncoder(os.environ['RERANKER_MODEL'])" \
    && chmod -R a+rX /opt/hf_cache
ENV HF_HUB_OFFLINE=1

# Copy the app code (after dependencies so code edits don't reinstall packages), then drop root
COPY --chown=appuser:appuser . .
USER appuser

# Document that the app listens on port 8000 (publishing is done with -p)
EXPOSE 8000

# Startup is slow (Milvus, PyTorch import, reranker load): about 2-3 minutes on a laptop, so allow 5
# Docker marks the container unhealthy if /api/health stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

# Production command (no reload). For local dev with reload run uvicorn with --reload (see README.md).
CMD ["uvicorn", "fda_api_main:app", "--host", "0.0.0.0", "--port", "8000"]
