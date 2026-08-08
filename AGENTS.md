# Repository guidance

- Preserve the independent, domain-agnostic `query + document` API.
- Never log or persist input text, bearer credentials, or un-hashed cache inputs.
- Keep model revision immutable and use a single model instance per container.
- Redis failures must degrade caching, not readiness or inference.
- Add tests for contract, stable ordering and error behavior with every API change.
- Run Ruff, mypy and pytest before handoff; do not download the production model in unit tests.

