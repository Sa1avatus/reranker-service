"""Tests for evidence-focused reranking behaviour.

The reranker must rank documents by semantic relevance to a query
(atomic claim) without making business-domain decisions about match
percentage, requirement satisfaction, or candidate suitability.

These tests exercise multilingual queries, false-positive resilience,
top-k aliasing, metadata preservation, edge cases, and stable ordering.
"""

import pytest
from pydantic import ValidationError

from reranker_service.schemas import RerankRequest

# ── helpers ──────────────────────────────────────────────────────────────────


def _rerank(client, auth, *, query: str, documents: list[dict], **kwargs):
    """Convenience wrapper that posts to /v1/rerank and returns the JSON body."""
    payload: dict = {"query": query, "documents": documents, **kwargs}
    response = client.post("/v1/rerank", json=payload, headers=auth)
    return response


def _score_map(response_json: dict) -> dict[str, float]:
    """Return {document_id: score} from a rerank response."""
    return {r["id"]: r["score"] for r in response_json["results"]}


# ── 1. Multilingual: RU query → EN evidence ─────────────────────────────────


class TestMultilingualReranking:
    """Cross-language evidence should rank higher than irrelevant evidence.

    The mock test model uses word-overlap scoring, so shared Latin-script
    technical terms (e.g. "LLM", "inference", "reranking") are needed for
    discrimination.  A real multilingual cross-encoder like
    ``BAAI/bge-reranker-v2-m3`` performs genuine cross-language semantic
    matching without requiring shared vocabulary.
    """

    def test_ru_query_en_evidence(self, client, auth):
        """Shared technical terms enable cross-language matching.
        A Russian claim about LLM inference should rank English evidence
        about LLM deployment higher than unrelated text."""
        response = _rerank(
            client,
            auth,
            query="Опыт работы с LLM и inference",
            documents=[
                {
                    "id": "unrelated",
                    "text": "Professional pastry chef with 10 years experience.",
                },
                {
                    "id": "relevant-en",
                    "text": "LLM inference deployment using Ollama and GPU servers.",
                },
                {
                    "id": "somewhat",
                    "text": "Worked as a software engineer on web applications.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["relevant-en"] > scores["unrelated"]
        assert scores["relevant-en"] > scores["somewhat"]

    def test_en_query_ru_evidence(self, client, auth):
        """An English claim about reranking should rank Russian evidence about
        cross-encoder reranking higher than unrelated Russian text."""
        response = _rerank(
            client,
            auth,
            query="Experience with cross-encoder reranking service",
            documents=[
                {
                    "id": "unrelated-ru",
                    "text": "Бухгалтер с опытом работы 5 лет в строительной компании.",
                },
                {
                    "id": "relevant-ru",
                    "text": "Реализовал cross-encoder reranking service на Python.",
                },
                {
                    "id": "partial-ru",
                    "text": "Опыт разработки REST API на FastAPI и Python.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["relevant-ru"] > scores["unrelated-ru"]

    def test_ru_query_ru_evidence_same_language(self, client, auth):
        """Same-language relevance should work normally."""
        response = _rerank(
            client,
            auth,
            query="Практический опыт работы с Kubernetes",
            documents=[
                {
                    "id": "k8s-ru",
                    "text": "Развёртывание микросервисов в Kubernetes кластере.",
                },
                {
                    "id": "unrelated-ru",
                    "text": "Занимался продвижением сайтов в социальных сетях.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["k8s-ru"] > scores["unrelated-ru"]

    def test_en_query_en_evidence_same_language(self, client, auth):
        """Same-language relevance should work normally."""
        response = _rerank(
            client,
            auth,
            query="Production Kubernetes experience",
            documents=[
                {
                    "id": "k8s-en",
                    "text": "Managed production Kubernetes clusters for 3 years.",
                },
                {
                    "id": "unrelated-en",
                    "text": "Catering manager for corporate events.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["k8s-en"] > scores["unrelated-en"]


# ── 2. False-positive resilience ────────────────────────────────────────────


class TestFalsePositiveResilience:
    """The reranker may give non-zero scores to thematically related but
    semantically different documents. It must NOT attempt to decide whether
    the evidence actually satisfies a requirement — that is the job of the
    downstream Evidence Evaluator.

    With the mock test model (word-overlap), documents sharing more query
    words score higher.  A real cross-encoder additionally captures semantic
    context and can distinguish "read about X" from "built with X".
    """

    def test_topical_but_not_evidence_gets_lower_score(self, client, auth):
        """Documents with higher word overlap score higher.  The reranker
        returns a relevance score; the Evidence Evaluator decides whether
        the evidence satisfies the claim."""
        response = _rerank(
            client,
            auth,
            query="Практический опыт vLLM inference",
            documents=[
                {
                    "id": "actual-evidence",
                    "text": "Запускал vLLM inference сервер для обслуживания LLM запросов.",
                },
                {
                    "id": "topical-not-evidence",
                    "text": "Изучал возможности vLLM, но в проектах не использовал.",
                },
                {
                    "id": "unrelated",
                    "text": "Профессиональный шеф-повар с 10-летним стажем.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        # Both "actual-evidence" and "topical-not-evidence" share "vLLM" with query.
        # actual-evidence also shares "inference" → higher score.
        assert scores["actual-evidence"] > scores["topical-not-evidence"]
        # Topical text shares "vLLM" → ranks above completely unrelated text.
        assert scores["topical-not-evidence"] > scores["unrelated"]

    def test_word_overlap_proportional_scoring(self, client, auth):
        """More overlapping query words → higher score.  This confirms the
        reranker returns proportional relevance, not binary accept/reject."""
        response = _rerank(
            client,
            auth,
            query="Python FastAPI backend REST API",
            documents=[
                {
                    "id": "high-overlap",
                    "text": "Built Python FastAPI backend REST API services.",
                },
                {
                    "id": "low-overlap",
                    "text": "Python scripting for data analysis.",
                },
                {
                    "id": "no-overlap",
                    "text": "Managed supply chain logistics for retail distribution.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["high-overlap"] > scores["low-overlap"]
        assert scores["low-overlap"] > scores["no-overlap"]


# ── 3. top_k alias for top_n ────────────────────────────────────────────────


class TestTopKAlias:
    """The API must accept 'top_k' as an alias for 'top_n' for backward
    compatibility with consumers that use that naming convention."""

    def test_top_k_alias_accepted(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="Kubernetes experience",
            documents=[
                {"id": "a", "text": "Production Kubernetes operations."},
                {"id": "b", "text": "SQL database administration."},
                {"id": "c", "text": "Kubernetes cluster management."},
            ],
            top_k=2,
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2

    def test_top_n_still_accepted(self, client, auth):
        """Existing clients using top_n must not break."""
        response = _rerank(
            client,
            auth,
            query="Kubernetes experience",
            documents=[
                {"id": "a", "text": "Production Kubernetes operations."},
                {"id": "b", "text": "SQL database administration."},
                {"id": "c", "text": "Kubernetes cluster management."},
            ],
            top_n=1,
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_top_k_and_top_n_both_present_rejects(self, client, auth):
        """Cannot specify both top_n and top_k simultaneously."""
        response = client.post(
            "/v1/rerank",
            json={
                "query": "test",
                "documents": [{"id": "1", "text": "test"}],
                "top_n": 1,
                "top_k": 1,
            },
            headers=auth,
        )
        # Should reject — ambiguous
        assert response.status_code == 422


# ── 4. Document ID preservation ─────────────────────────────────────────────


class TestDocumentIdPreservation:
    """Document IDs must pass through unchanged."""

    def test_ids_preserved_in_results(self, client, auth):
        doc_ids = ["fact-101", "resume-55-chunk-12", "uuid-style-id-abc"]
        response = _rerank(
            client,
            auth,
            query="Python experience",
            documents=[
                {"id": doc_id, "text": f"Document about Python for {doc_id}."}
                for doc_id in doc_ids
            ],
        )
        assert response.status_code == 200
        result_ids = [r["id"] for r in response.json()["results"]]
        assert set(result_ids) == set(doc_ids)

    def test_unicode_ids_preserved(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="test query",
            documents=[
                {"id": "профиль-42", "text": "Тестовый документ."},
                {"id": "résulté-7", "text": "Another test document."},
            ],
        )
        assert response.status_code == 200
        result_ids = {r["id"] for r in response.json()["results"]}
        assert result_ids == {"профиль-42", "résulté-7"}


# ── 5. Metadata preservation ────────────────────────────────────────────────


class TestMetadataPreservation:
    """Metadata must be returned unchanged regardless of return_documents."""

    def test_metadata_preserved_with_return_documents_true(self, client, auth):
        metadata = {"source_type": "fact", "source_id": "101", "user_id": "user-1"}
        response = _rerank(
            client,
            auth,
            query="test claim",
            documents=[
                {"id": "d1", "text": "test evidence", "metadata": metadata},
            ],
            return_documents=True,
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["metadata"] == metadata
        assert result["text"] == "test evidence"

    def test_metadata_preserved_when_return_documents_false(self, client, auth):
        """Metadata must always be returned, even when return_documents=false."""
        metadata = {"source_type": "resume", "chunk_id": "55-12"}
        response = _rerank(
            client,
            auth,
            query="test claim",
            documents=[
                {"id": "d1", "text": "test evidence text", "metadata": metadata},
            ],
            return_documents=False,
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        assert result["metadata"] == metadata
        # text should be omitted when return_documents=False
        assert result["text"] is None

    def test_empty_metadata_preserved(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="test",
            documents=[
                {"id": "d1", "text": "test document"},
            ],
        )
        assert response.status_code == 200
        result = response.json()["results"][0]
        # Default metadata is empty dict
        assert result["metadata"] == {}


# ── 6. Empty and boundary cases ─────────────────────────────────────────────


class TestEdgeCases:
    """Error handling and boundary conditions."""

    def test_empty_documents_rejected(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="some query",
            documents=[],
        )
        assert response.status_code == 422

    def test_empty_query_rejected(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="",
            documents=[{"id": "1", "text": "some text"}],
        )
        assert response.status_code == 422

    def test_single_document_returns_single_result(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="Docker experience",
            documents=[{"id": "only", "text": "Docker container deployment."}],
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == "only"
        assert results[0]["rank"] == 1


# ── 7. Batch request ────────────────────────────────────────────────────────


class TestBatchRequests:
    """Batch reranking must separate results per request."""

    def test_batch_preserves_per_request_results(self, client, auth):
        response = client.post(
            "/v1/rerank/batch",
            json={
                "requests": [
                    {
                        "query": "Python FastAPI experience",
                        "documents": [
                            {"id": "py-1", "text": "Built FastAPI REST services."},
                            {"id": "py-2", "text": "Java Spring Boot developer."},
                        ],
                    },
                    {
                        "query": "Data science experience",
                        "documents": [
                            {"id": "ds-1", "text": "Machine learning model training."},
                            {"id": "ds-2", "text": "Graphic designer for print media."},
                        ],
                    },
                ]
            },
            headers=auth,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["responses"]) == 2
        assert data["total_pairs"] == 4
        # Each response should have its own results
        first_ids = {r["id"] for r in data["responses"][0]["results"]}
        second_ids = {r["id"] for r in data["responses"][1]["results"]}
        assert first_ids == {"py-1", "py-2"}
        assert second_ids == {"ds-1", "ds-2"}

    def test_batch_too_large_rejected(self, client, auth, settings):
        requests = [
            {
                "query": f"query {i}",
                "documents": [{"id": str(i), "text": f"doc {i}"}],
            }
            for i in range(settings.max_batch_requests + 1)
        ]
        response = client.post(
            "/v1/rerank/batch",
            json={"requests": requests},
            headers=auth,
        )
        assert response.status_code == 413


# ── 8. Identical / near-identical documents ─────────────────────────────────


class TestDuplicateDocuments:
    """Identical or near-identical documents should get the same score
    and preserve stable input order."""

    def test_identical_documents_get_same_score_and_stable_order(self, client, auth):
        text = "Production Kubernetes cluster management."
        response = _rerank(
            client,
            auth,
            query="Kubernetes experience",
            documents=[
                {"id": "first", "text": text},
                {"id": "second", "text": text},
                {"id": "third", "text": text},
            ],
        )
        assert response.status_code == 200
        results = response.json()["results"]
        scores = [r["score"] for r in results]
        # All scores should be identical
        assert scores[0] == scores[1] == scores[2]
        # Stable sort: input order preserved for equal scores
        assert [r["id"] for r in results] == ["first", "second", "third"]

    def test_near_identical_documents_similar_scores(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="Python web development",
            documents=[
                {
                    "id": "exact",
                    "text": "Developed REST APIs using Python and FastAPI framework.",
                },
                {
                    "id": "near",
                    "text": "Developed REST APIs using Python and the FastAPI framework.",
                },
                {
                    "id": "different",
                    "text": "Managed supply chain logistics for retail distribution.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        # Near-identical should have very similar scores
        assert abs(scores["exact"] - scores["near"]) < abs(
            scores["exact"] - scores["different"]
        )


# ── 9. Irrelevant evidence ──────────────────────────────────────────────────


class TestIrrelevantEvidence:
    """Completely irrelevant evidence should receive lower scores."""

    def test_unrelated_documents_score_lower(self, client, auth):
        response = _rerank(
            client,
            auth,
            query="Machine learning model training with PyTorch",
            documents=[
                {
                    "id": "ml-relevant",
                    "text": "Trained deep learning models using PyTorch and GPU clusters.",
                },
                {
                    "id": "cooking",
                    "text": "Italian pasta recipes with homemade tomato sauce.",
                },
                {
                    "id": "sports",
                    "text": "Football match results and player statistics for 2024.",
                },
            ],
        )
        assert response.status_code == 200
        scores = _score_map(response.json())
        assert scores["ml-relevant"] > scores["cooking"]
        assert scores["ml-relevant"] > scores["sports"]


# ── 10. Backward compatibility ──────────────────────────────────────────────


class TestBackwardCompatibility:
    """Existing API contracts must not break."""

    def test_response_shape_unchanged(self, client, auth, payload):
        response = client.post("/v1/rerank", json=payload, headers=auth)
        assert response.status_code == 200
        data = response.json()
        # Required top-level fields
        for field in [
            "request_id",
            "model",
            "model_revision",
            "device",
            "requested_revision",
            "resolved_revision",
            "backend",
            "rerank_mode",
            "active_provider",
            "results",
            "usage",
        ]:
            assert field in data, f"Missing response field: {field}"
        # Required result fields
        result = data["results"][0]
        for field in [
            "id",
            "score",
            "normalized_score",
            "rank",
            "text",
            "metadata",
            "token_count",
            "truncated",
            "cache_hit",
        ]:
            assert field in result, f"Missing result field: {field}"
        # Required usage fields
        usage = data["usage"]
        for field in [
            "documents_received",
            "documents_scored",
            "cache_hits",
            "latency_ms",
        ]:
            assert field in usage, f"Missing usage field: {field}"

    def test_score_is_numeric(self, client, auth, payload):
        response = client.post("/v1/rerank", json=payload, headers=auth)
        assert response.status_code == 200
        for result in response.json()["results"]:
            assert isinstance(result["score"], int | float)

    def test_normalized_score_in_range_when_present(self, client, auth, payload):
        response = client.post("/v1/rerank", json=payload, headers=auth)
        assert response.status_code == 200
        for result in response.json()["results"]:
            ns = result["normalized_score"]
            if ns is not None:
                assert 0 <= ns <= 1

    def test_ranks_are_sequential_starting_from_one(self, client, auth, payload):
        response = client.post("/v1/rerank", json=payload, headers=auth)
        assert response.status_code == 200
        ranks = [r["rank"] for r in response.json()["results"]]
        assert ranks == list(range(1, len(ranks) + 1))


# ── 11. Schema-level tests ──────────────────────────────────────────────────


class TestSchemaValidation:
    """Direct Pydantic model tests for the top_k alias."""

    def test_top_k_field_accepted_by_schema(self):
        request = RerankRequest.model_validate(
            {
                "query": "test",
                "documents": [{"id": "1", "text": "test"}],
                "top_k": 5,
            }
        )
        assert request.top_n == 5

    def test_top_n_field_accepted_by_schema(self):
        request = RerankRequest.model_validate(
            {
                "query": "test",
                "documents": [{"id": "1", "text": "test"}],
                "top_n": 3,
            }
        )
        assert request.top_n == 3

    def test_schema_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            RerankRequest.model_validate(
                {"query": "test", "documents": [{"id": "1", "text": ""}]}
            )

    def test_schema_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            RerankRequest.model_validate(
                {"query": "", "documents": [{"id": "1", "text": "test"}]}
            )

    def test_schema_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            RerankRequest.model_validate(
                {
                    "query": "test",
                    "documents": [{"id": "1", "text": "test"}],
                    "unknown_field": "value",
                }
            )

    def test_default_metadata_is_empty_dict(self):
        request = RerankRequest.model_validate(
            {"query": "test", "documents": [{"id": "1", "text": "test"}]}
        )
        assert request.documents[0].metadata == {}
