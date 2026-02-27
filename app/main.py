"""
FastAPI Application
- /ingest: document ingestion
- /search: hybrid retrieval
- /summarize: LLM summary generation
- /evaluate: controlled experiments
- /agent: LangChain agent endpoint
- /dashboard: usage stats
- /health: health check
"""

import time
import logging
from contextlib import asynccontextmanager
from collections import defaultdict
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SummarizeRequest,
    SummarizeResponse,
    ThemeSummary,
    EvaluationRequest,
    EvaluationResponse,
    EvaluationResult,
    UsageStats,
    HealthResponse,
)
from app.pipeline.ingestion import IngestionPipeline, EmbeddingPipeline
from app.pipeline.indexing import FAISSIndex, ChromaStore
from app.core.retrieval import HybridRetriever
from app.core.generation import SummaryGenerator, SurveyAnalysisAgent
from app.core.evaluation import ExperimentRunner, TFIDFBaseline, compute_rouge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Global state (in production use dependency injection)
# ──────────────────────────────────────────────
embedder = EmbeddingPipeline()
faiss_index = FAISSIndex()
chroma_store = ChromaStore()
retriever = HybridRetriever(faiss_index, chroma_store, embedder)
ingestion_pipeline = IngestionPipeline(embedder)
generator = SummaryGenerator()
tfidf_baseline = TFIDFBaseline()
experiment_runner = ExperimentRunner(retriever, generator, tfidf_baseline)
agent = SurveyAnalysisAgent(retriever)

# Usage tracking
usage_stats: Dict = defaultdict(int)
request_log: List[Dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted indices on startup."""
    logger.info("Starting RAG Survey Summarizer...")
    faiss_index.load()
    yield
    faiss_index.save()
    logger.info("Shutdown complete")


app = FastAPI(
    title=settings.APP_TITLE,
    description="LLM-powered RAG system for survey response summarization",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Middleware: request logging & usage tracking
# ──────────────────────────────────────────────
@app.middleware("http")
async def track_usage(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - start) * 1000

    endpoint = request.url.path
    usage_stats[f"endpoint:{endpoint}"] += 1
    usage_stats["total_requests"] += 1
    usage_stats[f"latency_sum:{endpoint}"] += latency_ms

    request_log.append(
        {
            "endpoint": endpoint,
            "method": request.method,
            "latency_ms": round(latency_ms, 2),
            "status": response.status_code,
        }
    )
    if len(request_log) > 1000:
        request_log.pop(0)

    return response


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        faiss_docs=faiss_index.total,
        chroma_docs=chroma_store.total,
        schema_version=settings.SCHEMA_VERSION,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Data"])
async def ingest(request: IngestRequest):
    """
    Ingest survey documents:
    1. Chunk text with overlap
    2. Generate sentence-transformer embeddings
    3. Index in FAISS IVF + ChromaDB
    """
    start = time.perf_counter()

    docs = [doc.dict() for doc in request.documents]
    chunks, ingested, failed = ingestion_pipeline.process(
        docs, batch_size=request.batch_size
    )

    if chunks:
        chunk_ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings_array = __import__("numpy").array([c.embedding for c in chunks])

        faiss_index.add(embeddings_array, chunk_ids, texts)
        chroma_store.add(chunks)
        retriever.build_bm25(chunk_ids, texts)
        tfidf_baseline.fit(chunk_ids, texts)

    duration_ms = (time.perf_counter() - start) * 1000
    usage_stats["total_documents"] += ingested

    return IngestResponse(
        ingested=ingested,
        failed=failed,
        schema_version=settings.SCHEMA_VERSION,
        duration_ms=round(duration_ms, 2),
    )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search(request: SearchRequest):
    """
    Retrieve relevant survey chunks.
    Modes: sparse (BM25) | dense (FAISS) | hybrid (RRF fusion)
    """
    if faiss_index.total == 0 and retriever.bm25.total == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. POST to /ingest first."
        )

    chunks, latency_ms = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode.value,
        filters=request.filters,
    )

    usage_stats[f"mode:{request.mode.value}"] += 1

    results = [
        SearchResult(
            id=c.id,
            text=c.text,
            score=round(c.score, 4),
            retrieval_mode=c.retrieval_mode,
            metadata=c.metadata or {},
        )
        for c in chunks
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        mode=request.mode,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/summarize", response_model=SummarizeResponse, tags=["Generation"])
async def summarize(request: SummarizeRequest):
    """
    Full RAG pipeline: retrieve + generate executive summary with theme extraction.
    Includes ROUGE evaluation against retrieved context as self-reference.
    """
    if faiss_index.total == 0 and retriever.bm25.total == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. POST to /ingest first."
        )

    chunks, retrieval_latency = retriever.retrieve(
        query=request.query, top_k=request.top_k, mode=request.mode.value
    )

    if not chunks:
        raise HTTPException(
            status_code=404, detail="No relevant responses found for this query."
        )

    gen_result = generator.generate(
        query=request.query,
        chunks=chunks,
        max_themes=request.max_themes,
        use_strong_model=request.use_strong_model,
    )

    # ROUGE self-evaluation vs retrieved context
    source_texts = [c.text for c in chunks[:3]]
    rouge_scores = compute_rouge(gen_result.get("executive_summary", ""), source_texts)

    themes = [
        ThemeSummary(
            theme=t.get("theme", ""),
            summary=t.get("summary", ""),
            supporting_responses=t.get("supporting_responses", []),
            response_count=t.get("response_count", 0),
            confidence=t.get("confidence", 0.5),
        )
        for t in gen_result.get("themes", [])
    ]

    total_latency = retrieval_latency + gen_result.get("latency_ms", 0)

    return SummarizeResponse(
        query=request.query,
        executive_summary=gen_result.get("executive_summary", ""),
        themes=themes,
        total_responses_analyzed=len(chunks),
        retrieval_mode=request.mode.value,
        model_used=gen_result.get("model", settings.OPENAI_MODEL),
        latency_ms=round(total_latency, 2),
        rouge_scores=rouge_scores,
    )


@app.post("/evaluate", response_model=EvaluationResponse, tags=["Evaluation"])
async def evaluate(request: EvaluationRequest):
    """
    Run controlled experiment comparing sparse vs dense vs hybrid vs TF-IDF baseline.
    Reports ROUGE scores, theme accuracy, hallucination score per mode.
    """
    experiment = experiment_runner.run_all_modes(
        query=request.query,
        ground_truth_themes=request.ground_truth_themes,
        top_k=request.top_k,
    )

    results = []
    for mode_str, r in experiment["results"].items():
        results.append(
            EvaluationResult(
                mode=mode_str,
                rouge_1=r.get("rouge_1", 0),
                rouge_2=r.get("rouge_2", 0),
                rouge_l=r.get("rouge_l", 0),
                theme_detection_accuracy=r.get("theme_accuracy", 0),
                hallucination_score=r.get("hallucination_score", 0),
                latency_ms=r.get("latency_ms", 0),
            )
        )

    return EvaluationResponse(
        query=request.query, results=results, best_mode=experiment["best_mode"]
    )


@app.post("/agent", tags=["Agent"])
async def run_agent(query: str):
    """
    LangChain agent endpoint with tool-calling.
    Agent can: semantic search, SQL query, multi-step reasoning.
    """
    result = agent.run(query)
    return {
        "query": query,
        "answer": result["answer"],
        "latency_ms": result["latency_ms"],
    }


@app.get("/dashboard", response_model=UsageStats, tags=["Monitoring"])
async def dashboard():
    """Usage statistics dashboard."""
    total = usage_stats.get("total_requests", 0)
    avg_latency = 0.0
    if total > 0:
        total_latency = sum(
            v for k, v in usage_stats.items() if k.startswith("latency_sum:")
        )
        avg_latency = total_latency / total

    requests_by_mode = {
        k.replace("mode:", ""): v
        for k, v in usage_stats.items()
        if k.startswith("mode:")
    }

    requests_by_endpoint = {
        k.replace("endpoint:", ""): v
        for k, v in usage_stats.items()
        if k.startswith("endpoint:")
    }

    recent_queries = [
        r["endpoint"] for r in request_log[-10:] if "search" in r["endpoint"]
    ]

    return UsageStats(
        total_requests=total,
        total_documents=usage_stats.get("total_documents", 0),
        avg_latency_ms=round(avg_latency, 2),
        requests_by_mode=requests_by_mode,
        requests_by_endpoint=requests_by_endpoint,
        top_queries=recent_queries,
    )
