# syntax=docker/dockerfile:1.7
FROM python:3.12.4-slim-bookworm AS cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN groupadd -r reranker && useradd -r -g reranker -d /app reranker
WORKDIR /app
COPY pyproject.toml README.md ./
COPY reranker_service ./reranker_service
RUN pip install --no-cache-dir .
RUN mkdir -p /models && chown -R reranker:reranker /app /models
USER reranker
EXPOSE 8200
CMD ["uvicorn","reranker_service.main:app","--host","0.0.0.0","--port","8200","--workers","1"]

FROM cpu AS cuda
USER root
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1
USER reranker

