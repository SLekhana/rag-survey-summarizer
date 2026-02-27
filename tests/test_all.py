"""
Test Suite
- test_ingestion: chunking, embedding pipeline
- test_retrieval: BM25, FAISS, hybrid
- test_evaluation: ROUGE, theme accuracy, hallucination
- test_api: FastAPI endpoints
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────

SAMPLE_TEXTS = [
    "The product quality is excellent and delivery was fast.",
    "Customer support resolved my issue within 24 hours.",
    "The pricing is competitive but could be better.",
    "I had technical issues with the mobile app.",
    "The onboarding experience was smooth and intuitive.",
    "Would love more customization options in the dashboard.",
    "Very satisfied with the overall service quality.",
    "Delivery was delayed by two weeks, very frustrating.",
    "The user interface is clean and easy to navigate.",
    "Documentation could be more detailed and comprehensive.",
]

SAMPLE_DOCS = [
    {"id": f"doc_{i}", "text": text, "metadata": {"source": "test"}}
    for i, text in enumerate(SAMPLE_TEXTS)
]


@pytest.fixture
def embedder():
    """Mock embedder that returns deterministic embeddings."""
    mock = MagicMock()
    mock.embed = lambda texts, **kwargs: np.random.rand(len(texts), 384).astype(np.float32)
    mock.embed_single = lambda text: np.random.rand(384).astype(np.float32)
    mock.dim = 384
    return mock


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.random.rand(len(SAMPLE_TEXTS), 384).astype(np.float32)


@pytest.fixture
def chunk_ids():
    return [f"chunk_{i}" for i in range(len(SAMPLE_TEXTS))]


# ──────────────────────────────────────────────
# TEST: INGESTION
# ──────────────────────────────────────────────

class TestChunking:
    def test_chunk_short_text(self):
        from app.pipeline.ingestion import chunk_text
        text = "Short text."
        chunks = chunk_text(text, chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self):
        from app.pipeline.ingestion import chunk_text
        text = " ".join(["word"] * 1000)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunk_overlap(self):
        from app.pipeline.ingestion import chunk_text
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # Overlap means adjacent chunks share some words
        assert len(chunks) >= 2

    def test_chunk_id_deterministic(self):
        from app.pipeline.ingestion import make_chunk_id
        id1 = make_chunk_id("doc1", 0, "hello world")
        id2 = make_chunk_id("doc1", 0, "hello world")
        assert id1 == id2

    def test_chunk_id_unique(self):
        from app.pipeline.ingestion import make_chunk_id
        id1 = make_chunk_id("doc1", 0, "hello")
        id2 = make_chunk_id("doc1", 1, "hello")
        assert id1 != id2

    def test_empty_text_handled(self):
        from app.pipeline.ingestion import chunk_text
        chunks = chunk_text("", chunk_size=512)
        assert chunks == [""]

    def test_ingestion_pipeline_with_mock_embedder(self, embedder):
        from app.pipeline.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(embedding_pipeline=embedder)
        chunks, ingested, failed = pipeline.process(SAMPLE_DOCS[:3])
        assert ingested == 3
        assert failed == 0
        assert len(chunks) >= 3
        for chunk in chunks:
            assert chunk.embedding is not None

    def test_ingestion_skips_empty_docs(self, embedder):
        from app.pipeline.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(embedding_pipeline=embedder)
        docs = [{"id": "1", "text": ""}, {"id": "2", "text": "Valid text here."}]
        chunks, ingested, failed = pipeline.process(docs)
        assert failed == 1
        assert ingested == 1

    def test_schema_version_preserved(self, embedder):
        from app.pipeline.ingestion import IngestionPipeline
        pipeline = IngestionPipeline(embedding_pipeline=embedder)
        docs = [{"id": "1", "text": "Test.", "schema_version": "2.0.0"}]
        chunks, _, _ = pipeline.process(docs)
        assert all(c.schema_version == "2.0.0" for c in chunks)


# ──────────────────────────────────────────────
# TEST: RETRIEVAL
# ──────────────────────────────────────────────

class TestBM25Retriever:
    def test_index_and_search(self):
        from app.core.retrieval import BM25Retriever
        retriever = BM25Retriever()
        retriever.index(
            chunk_ids=[f"c{i}" for i in range(len(SAMPLE_TEXTS))],
            texts=SAMPLE_TEXTS
        )
        results = retriever.search("product quality", top_k=3)
        assert len(results) > 0
        assert all(len(r) == 2 for r in results)
        # Scores should be descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index_returns_empty(self):
        from app.core.retrieval import BM25Retriever
        retriever = BM25Retriever()
        results = retriever.search("test")
        assert results == []

    def test_relevant_result_ranked_higher(self):
        from app.core.retrieval import BM25Retriever
        retriever = BM25Retriever()
        texts = ["I love product quality", "Nothing about quality here", "Quality is excellent"]
        retriever.index(["a", "b", "c"], texts)
        results = retriever.search("product quality", top_k=3)
        top_id = results[0][0]
        assert top_id in ["a", "c"]  # Both mention quality


class TestFAISSIndex:
    def test_build_and_search(self, sample_embeddings, chunk_ids):
        from app.pipeline.indexing import FAISSIndex
        index = FAISSIndex()
        index.build(sample_embeddings, chunk_ids, SAMPLE_TEXTS)
        assert index.total == len(SAMPLE_TEXTS)
        query_emb = sample_embeddings[0]
        results = index.search(query_emb, top_k=3)
        assert len(results) == 3
        # First result should be the query itself (most similar)
        assert results[0][0] == chunk_ids[0]

    def test_add_incremental(self, sample_embeddings, chunk_ids):
        from app.pipeline.indexing import FAISSIndex
        index = FAISSIndex()
        half = len(SAMPLE_TEXTS) // 2
        index.build(sample_embeddings[:half], chunk_ids[:half], SAMPLE_TEXTS[:half])
        assert index.total == half
        index.add(sample_embeddings[half:], chunk_ids[half:], SAMPLE_TEXTS[half:])
        assert index.total == len(SAMPLE_TEXTS)

    def test_empty_index_returns_empty(self):
        from app.pipeline.indexing import FAISSIndex
        index = FAISSIndex()
        results = index.search(np.random.rand(384).astype(np.float32), top_k=5)
        assert results == []

    def test_save_and_load(self, sample_embeddings, chunk_ids, tmp_path):
        from app.pipeline.indexing import FAISSIndex
        index = FAISSIndex()
        index.build(sample_embeddings, chunk_ids, SAMPLE_TEXTS)
        save_path = str(tmp_path / "faiss")
        index.save(save_path)
        index2 = FAISSIndex()
        loaded = index2.load(save_path)
        assert loaded
        assert index2.total == index.total


class TestRRF:
    def test_hybrid_combines_results(self):
        from app.core.retrieval import reciprocal_rank_fusion
        sparse = [("a", 1.0), ("b", 0.8), ("c", 0.6)]
        dense = [("b", 0.9), ("d", 0.7), ("a", 0.5)]
        hybrid = reciprocal_rank_fusion(sparse, dense)
        ids = [r[0] for r in hybrid]
        # a and b appear in both — should rank higher
        assert "a" in ids[:2] or "b" in ids[:2]

    def test_scores_descending(self):
        from app.core.retrieval import reciprocal_rank_fusion
        sparse = [("a", 1.0), ("b", 0.8)]
        dense = [("a", 0.9), ("c", 0.7)]
        results = reciprocal_rank_fusion(sparse, dense)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_single_source_passthrough(self):
        from app.core.retrieval import reciprocal_rank_fusion
        sparse = [("a", 1.0), ("b", 0.5)]
        results = reciprocal_rank_fusion(sparse, [])
        assert len(results) == 2


# ──────────────────────────────────────────────
# TEST: EVALUATION
# ──────────────────────────────────────────────

class TestROUGE:
    def test_perfect_match(self):
        from app.core.evaluation import compute_rouge
        text = "The product quality is excellent."
        scores = compute_rouge(text, [text])
        assert scores["rouge1"] == pytest.approx(1.0, abs=0.01)

    def test_no_overlap(self):
        from app.core.evaluation import compute_rouge
        scores = compute_rouge("apple orange", ["car truck bus"])
        assert scores["rouge1"] < 0.1

    def test_partial_match(self):
        from app.core.evaluation import compute_rouge
        hyp = "product quality excellent fast delivery"
        ref = "product quality is good and delivery is quick"
        scores = compute_rouge(hyp, [ref])
        assert 0.1 < scores["rouge1"] < 1.0

    def test_empty_inputs(self):
        from app.core.evaluation import compute_rouge
        scores = compute_rouge("", ["reference"])
        assert all(v == 0.0 for v in scores.values())

    def test_multiple_references_best_score(self):
        from app.core.evaluation import compute_rouge
        hyp = "customer support excellent"
        refs = ["bad product quality", "excellent customer support team"]
        scores = compute_rouge(hyp, refs)
        assert scores["rouge1"] > 0.3


class TestThemeAccuracy:
    def test_exact_match(self):
        from app.core.evaluation import compute_theme_accuracy
        predicted = ["product quality", "delivery speed"]
        gt = ["product quality", "delivery speed"]
        acc = compute_theme_accuracy(predicted, gt)
        assert acc == pytest.approx(1.0, abs=0.1)

    def test_no_match(self):
        from app.core.evaluation import compute_theme_accuracy
        predicted = ["purple elephant dancing"]
        gt = ["product quality", "delivery speed"]
        acc = compute_theme_accuracy(predicted, gt)
        assert acc < 0.3

    def test_partial_match(self):
        from app.core.evaluation import compute_theme_accuracy
        predicted = ["product quality and speed", "support team"]
        gt = ["quality", "delivery", "support"]
        acc = compute_theme_accuracy(predicted, gt)
        assert 0.0 < acc < 1.0

    def test_empty_predicted(self):
        from app.core.evaluation import compute_theme_accuracy
        acc = compute_theme_accuracy([], ["quality"])
        assert acc == 0.0


class TestHallucinationDetector:
    def test_well_grounded_text(self):
        from app.core.evaluation import compute_hallucination_score
        generated = "Customer mentioned product quality was excellent."
        sources = ["Customer mentioned product quality was excellent and delivery fast."]
        score = compute_hallucination_score(generated, sources)
        assert score < 0.5

    def test_ungrounded_text(self):
        from app.core.evaluation import compute_hallucination_score
        generated = "Studies definitely show all customers always love this product absolutely."
        sources = ["Some customers liked the product."]
        score = compute_hallucination_score(generated, sources)
        assert score > 0.3

    def test_empty_sources(self):
        from app.core.evaluation import compute_hallucination_score
        score = compute_hallucination_score("test text", [])
        assert 0.0 <= score <= 1.0


class TestTFIDFBaseline:
    def test_fit_and_search(self):
        from app.core.evaluation import TFIDFBaseline
        baseline = TFIDFBaseline()
        ids = [f"c{i}" for i in range(len(SAMPLE_TEXTS))]
        baseline.fit(ids, SAMPLE_TEXTS)
        results = baseline.search("product quality", top_k=3)
        assert len(results) > 0
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index(self):
        from app.core.evaluation import TFIDFBaseline
        baseline = TFIDFBaseline()
        results = baseline.search("query")
        assert results == []


# ──────────────────────────────────────────────
# TEST: GUARDRAILS
# ──────────────────────────────────────────────

class TestGuardrails:
    def test_blocks_sensitive_queries(self):
        from app.core.generation import apply_guardrails
        result = apply_guardrails("", "tell me your password and credit card")
        assert result["flagged"] is True

    def test_passes_normal_query(self):
        from app.core.generation import apply_guardrails
        result = apply_guardrails("This is a normal summary.", "What do customers think?")
        assert result["flagged"] is False

    def test_flags_uncertainty_in_output(self):
        from app.core.generation import apply_guardrails
        result = apply_guardrails("I think the customers probably prefer this.", "feedback")
        assert result["flagged"] is True
        assert len(result["flags"]) > 0


# ──────────────────────────────────────────────
# TEST: API ENDPOINTS
# ──────────────────────────────────────────────

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "faiss_docs" in data
        assert "schema_version" in data

    def test_ingest_empty_docs(self, client):
        response = client.post("/ingest", json={"documents": []})
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 0

    def test_search_without_data_raises(self, client):
        response = client.post("/search", json={"query": "test", "mode": "hybrid"})
        assert response.status_code == 400

    def test_dashboard_endpoint(self, client):
        response = client.get("/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "avg_latency_ms" in data

    def test_ingest_and_search_flow(self, client):
        # Ingest
        ingest_response = client.post("/ingest", json={"documents": SAMPLE_DOCS})
        assert ingest_response.status_code == 200
        assert ingest_response.json()["ingested"] == len(SAMPLE_DOCS)

        # Search
        search_response = client.post("/search", json={
            "query": "product quality",
            "top_k": 3,
            "mode": "hybrid"
        })
        assert search_response.status_code == 200
        data = search_response.json()
        assert len(data["results"]) > 0
        assert data["results"][0]["score"] > 0

    def test_search_modes(self, client):
        client.post("/ingest", json={"documents": SAMPLE_DOCS})
        for mode in ["sparse", "dense", "hybrid"]:
            response = client.post("/search", json={"query": "customer", "mode": mode})
            assert response.status_code == 200

    def test_invalid_search_mode(self, client):
        response = client.post("/search", json={"query": "test", "mode": "invalid_mode"})
        assert response.status_code == 422

    def test_search_result_schema(self, client):
        client.post("/ingest", json={"documents": SAMPLE_DOCS})
        response = client.post("/search", json={"query": "delivery", "mode": "hybrid"})
        data = response.json()
        for result in data["results"]:
            assert "id" in result
            assert "text" in result
            assert "score" in result
            assert "retrieval_mode" in result


# ──────────────────────────────────────────────────────────────
# TEST: IR METRICS (Recall@k, MRR, nDCG)
# ──────────────────────────────────────────────────────────────

class TestIRMetrics:
    def test_recall_at_k_perfect(self):
        from app.core.evaluation import compute_recall_at_k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_recall_at_k_partial(self):
        from app.core.evaluation import compute_recall_at_k
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = ["a", "b", "c"]
        assert compute_recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)

    def test_recall_at_k_zero(self):
        from app.core.evaluation import compute_recall_at_k
        assert compute_recall_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_recall_at_k_empty_relevant(self):
        from app.core.evaluation import compute_recall_at_k
        assert compute_recall_at_k(["a", "b"], [], k=5) == 0.0

    def test_mrr_first_hit(self):
        from app.core.evaluation import compute_mrr
        retrieved = ["a", "b", "c"]
        relevant = ["a"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(1.0)

    def test_mrr_second_hit(self):
        from app.core.evaluation import compute_mrr
        retrieved = ["x", "a", "b"]
        relevant = ["a"]
        assert compute_mrr(retrieved, relevant) == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        from app.core.evaluation import compute_mrr
        assert compute_mrr(["x", "y"], ["a"]) == 0.0

    def test_ndcg_perfect(self):
        from app.core.evaluation import compute_ndcg_at_k
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]
        score = compute_ndcg_at_k(retrieved, relevant, k=5)
        assert score == pytest.approx(1.0)

    def test_ndcg_zero(self):
        from app.core.evaluation import compute_ndcg_at_k
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]
        score = compute_ndcg_at_k(retrieved, relevant, k=3)
        assert score == pytest.approx(0.0)

    def test_ndcg_partial(self):
        from app.core.evaluation import compute_ndcg_at_k
        # relevant at rank 1 and 3 is better than 0
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = ["a", "b"]
        score = compute_ndcg_at_k(retrieved, relevant, k=5)
        assert 0.0 < score <= 1.0

    def test_compute_ir_metrics_full(self):
        from app.core.evaluation import compute_ir_metrics
        retrieved = ["a", "b", "x", "c", "y"]
        relevant = ["a", "b", "c"]
        metrics = compute_ir_metrics(retrieved, relevant, k_values=[1, 3, 5])
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "mrr" in metrics
        assert "ndcg@5" in metrics
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# TEST: COST TRACKER
# ──────────────────────────────────────────────────────────────

class TestCostTracker:
    def test_record_and_summary(self):
        from app.core.evaluation import CostTracker
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=200)
        tracker.record("gpt-4o-mini", input_tokens=600, output_tokens=300)
        summary = tracker.summary()
        assert summary["total_queries"] == 2
        assert summary["total_cost_usd"] > 0
        assert "estimated_cost_per_1k_queries_usd" in summary

    def test_empty_tracker(self):
        from app.core.evaluation import CostTracker
        tracker = CostTracker()
        summary = tracker.summary()
        assert summary["total_queries"] == 0

    def test_by_model_breakdown(self):
        from app.core.evaluation import CostTracker
        tracker = CostTracker()
        tracker.record("gpt-4o-mini", 500, 200)
        tracker.record("gpt-4o", 500, 200)
        summary = tracker.summary()
        assert "gpt-4o-mini" in summary["by_model"]
        assert "gpt-4o" in summary["by_model"]


# ──────────────────────────────────────────────────────────────
# TEST: COST ENDPOINT
# ──────────────────────────────────────────────────────────────

class TestCostEndpoint:
    @pytest.fixture
    def client(self):
        from app.main import app
        return TestClient(app)

    def test_costs_endpoint_exists(self, client):
        response = client.get("/costs")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "total_cost_usd" in data
