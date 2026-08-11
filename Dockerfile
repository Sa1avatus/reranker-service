# syntax=docker/dockerfile:1.7

FROM python:3.12.4-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=600 \
    RERANKER_ARTIFACT_ROOT=/models/artifacts
RUN groupadd -r reranker && useradd -r -g reranker -d /app reranker
WORKDIR /app

FROM python-base AS python-runtime-base
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service

FROM python-runtime-base AS runtime-cpu
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install '.[onnx-cpu]'
RUN mkdir -p /models/artifacts && chown -R reranker:reranker /app /models
USER reranker
EXPOSE 8200
CMD ["python","-m","uvicorn","reranker_service.main:app","--host","0.0.0.0","--port","8200","--workers","1"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime-gpu
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=600 \
    RERANKER_ARTIFACT_ROOT=/models/artifacts \
    PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
RUN groupadd -r reranker && useradd -r -g reranker -d /app reranker
WORKDIR /app
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install '.[onnx-gpu]'
RUN mkdir -p /models/artifacts && chown -R reranker:reranker /app /models
USER reranker
EXPOSE 8200
CMD ["python","-m","uvicorn","reranker_service.main:app","--host","0.0.0.0","--port","8200","--workers","1"]

# Keep the large Torch/CUDA wheel in a stable layer shared by optional Torch backends.
FROM python-base AS cuda-torch
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1

FROM cuda-torch AS runtime-legacy
ENV HF_HOME=/models/hf-cache
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install '.[legacy]'
RUN mkdir -p /models/hf-cache && chown -R reranker:reranker /app /models
USER reranker
EXPOSE 8200
CMD ["python","-m","uvicorn","reranker_service.main:app","--host","0.0.0.0","--port","8200","--workers","1"]

FROM cuda-torch AS runtime-jina
ENV HF_HOME=/models/hf-cache
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install '.[jina]'
RUN mkdir -p /models/hf-cache && chown -R reranker:reranker /app /models
USER reranker
EXPOSE 8200
CMD ["python","-m","uvicorn","reranker_service.main:app","--host","0.0.0.0","--port","8200","--workers","1"]

FROM runtime-jina AS runtime-alibaba
USER root
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install transformers==4.39.1
USER reranker

FROM python:3.12.4-slim-bookworm AS exporter
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=600 \
    RERANKER_ARTIFACT_ROOT=/models/artifacts HF_HOME=/models/hf-cache
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
RUN groupadd -r reranker && useradd -r -g reranker -d /app reranker
WORKDIR /app
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install '.[exporter]'
RUN mkdir -p /models/artifacts /models/hf-cache && chown -R reranker:reranker /app /models
USER reranker
ENTRYPOINT ["reranker-export-onnx"]
CMD ["--help"]

# Backward-compatible build target names.
FROM runtime-cpu AS cpu
FROM runtime-gpu AS cuda
