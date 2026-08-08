.PHONY: up down logs test lint typecheck benchmark
up:
	docker compose up -d --build
down:
	docker compose down
logs:
	docker compose logs -f
test:
	pytest
lint:
	ruff check .
typecheck:
	mypy reranker_service
benchmark:
	python benchmarks/run.py --url http://localhost:8200

