# Development

Python 3.12 is required. Install the legacy development runtime with `python -m pip install -e '.[legacy,dev]'`, copy `.env.example` to `.env`, set two development-only secrets, and run `uvicorn reranker_service.main:app --reload --port 8200`. Install `.[onnx-cpu,dev]` for local ONNX runtime work, `.[jina,dev]` for Jina adapter work, or `.[exporter,dev]` for artifact export work. Alibaba development should use the pinned `runtime-alibaba` Docker target because its checkpoint requires the isolated Transformers 4.39.1 stack. Tests use deterministic fakes and need no model download. Run `pytest`, `ruff check .`, and `mypy reranker_service`.

Redis integration tests are opt-in: start `reranker-redis`, set `RUN_REDIS_INTEGRATION=1` and `REDIS_INTEGRATION_URL=redis://127.0.0.1:57379/15`, then run `pytest tests/test_redis_integration.py --no-cov`. `make integration-test` automates this on POSIX shells. The tests use database 15, remove their keys and ACL user, and never download the production model.

For the UI use Node 22 and pnpm: `cd web`, `corepack enable`, `pnpm install --frozen-lockfile`, `pnpm run dev`, `pnpm test`, and `pnpm run e2e`. Vite proxies API routes in development. English is the current locale; visible strings are centralized enough to move into a locale dictionary in the next UI iteration.
