from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class RetrievalMode(str, Enum):
    sparse = "sparse"  # BM25 only
    dense = "dense"  # FAISS only
    hybrid = "hybrid"  # BM25 + FAISS (default)


class SurveyDocument(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}
    schema_version: str = "1.0.0"


class IngestRequest(BaseModel):
    documents: List[SurveyDocument]
    batch_size: int = Field(default=32, ge=1, le=256)


class IngestResponse(BaseModel):
    ingested: int
    failed: int
    schema_version: str
    duration_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: RetrievalMode = RetrievalMode.hybrid
    filters: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    retrieval_mode: str
    metadata: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    mode: RetrievalMode
    latency_ms: float


class SummarizeRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: RetrievalMode = RetrievalMode.hybrid
    use_strong_model: bool = False
    max_themes: int = Field(default=5, ge=1, le=10)


class ThemeSummary(BaseModel):
    theme: str
    summary: str
    supporting_responses: List[str]
    response_count: int
    confidence: float


class SummarizeResponse(BaseModel):
    query: str
    executive_summary: str
    themes: List[ThemeSummary]
    total_responses_analyzed: int
    retrieval_mode: str
    model_used: str
    latency_ms: float
    rouge_scores: Optional[Dict[str, float]] = None


class EvaluationRequest(BaseModel):
    query: str
    ground_truth_themes: List[str]
    top_k: int = 10
    modes: List[RetrievalMode] = [
        RetrievalMode.sparse,
        RetrievalMode.dense,
        RetrievalMode.hybrid,
    ]


class EvaluationResult(BaseModel):
    mode: str
    rouge_1: float
    rouge_2: float
    rouge_l: float
    theme_detection_accuracy: float
    hallucination_score: float
    latency_ms: float


class EvaluationResponse(BaseModel):
    query: str
    results: List[EvaluationResult]
    best_mode: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UsageStats(BaseModel):
    total_requests: int
    total_documents: int
    avg_latency_ms: float
    requests_by_mode: Dict[str, int]
    requests_by_endpoint: Dict[str, int]
    top_queries: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    faiss_docs: int
    chroma_docs: int
    schema_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
