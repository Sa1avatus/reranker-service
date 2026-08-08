# Development

Python 3.12 is required. Install with `python -m pip install -e '.[dev]'`, copy `.env.example` to `.env`, set two development-only secrets, and run `uvicorn reranker_service.main:app --reload --port 8200`. Tests use a deterministic mock runtime and need no model download. Run `pytest`, `ruff check .`, and `mypy reranker_service`.

For the UI use Node 22: `cd web`, `npm install`, `npm run dev`, `npm test`, and `npm run e2e`. Vite proxies API routes in development. English is the current locale; visible strings are centralized enough to move into a locale dictionary in the next UI iteration.

