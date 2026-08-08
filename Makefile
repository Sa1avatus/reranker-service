.PHONY: up down logs test integration-test lint typecheck benchmark
up:
	docker compose up -d --build
down:
	docker compose down
logs:
	docker compose logs -f
test:
	pytest
integration-test:
	docker compose up -d reranker-redis
	RUN_REDIS_INTEGRATION=1 REDIS_INTEGRATION_URL=redis://127.0.0.1:57379/15 pytest tests/test_redis_integration.py --no-cov
	docker compose stop reranker-redis
lint:
	ruff check .
typecheck:
	mypy reranker_service
benchmark:
	python benchmarks/run.py --url http://localhost:8200
