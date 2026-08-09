# Repository guidance

- Preserve the independent, domain-agnostic `query + document` API.
- Never log or persist input text, bearer credentials, or un-hashed cache inputs.
- Keep model revision immutable and use a single model instance per container.
- Redis failures must degrade caching, not readiness or inference.
- Add tests for contract, stable ordering and error behavior with every API change.
- Run Ruff, mypy and pytest before handoff; do not download the production model in unit tests.
- Keep project onboarding documentation bilingual: `README.md` is the default English version and
  `README.ru.md` is the complete Russian translation. Link both language versions at the top,
  update them together, and keep their structure and facts synchronized.
- Include a Mermaid architecture diagram in both README language versions. Follow the
  `job-searching-assistant` style: a compact `flowchart LR` that shows users/clients, runtime
  services, model/cache ownership, and the principal request/data paths.
