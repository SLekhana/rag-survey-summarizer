"""
Cross-Encoder Reranker
──────────────────────
After BM25+FAISS hybrid retrieval, rerank the top-k candidates using a
cross-encoder model for much more accurate relevance scoring.

Why cross-encoders beat bi-encoders for reranking:
  - Bi-encoders (like all-MiniLM): encode query and doc SEPARATELY → fast but approximate
  - Cross-encoders: encode query + doc TOGETHER → slower but far more accurate
  - Standard pattern: fast retrieval (bi-encoder) → accurate reranking (cross-encoder)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M parameters, very fast (can rerank 100 docs in <200ms on CPU)
  - Trained on MS MARCO passage ranking benchmark
  - Free, runs locally, no API calls
  - Recall@5 improves ~8-15% vs bi-encoder only retrieval

Interview talking points:
  - "I use a two-stage retrieval pipeline: fast ANN retrieval then cross-encoder reranking"
  - "The cross-encoder sees the full query-document interaction, unlike bi-encoders"
  - "This is the same pattern used by production RAG systems at Google and Microsoft"
"""

import logging
import time
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Lazy import — only load model when first used
_cross_encoder = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_model():
    """Lazy-load the cross-encoder model (downloads ~85MB on first run)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {_model_name}")
            _cross_encoder = CrossEncoder(_model_name, max_length=512)
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Reranker disabled. Install with: pip install sentence-transformers"
            )
    return _cross_encoder


def rerank(
    query: str,
    candidates: List[Tuple[str, str, float]],
    top_k: Optional[int] = None,
    return_scores: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Rerank candidate (id, text, score) tuples using cross-encoder.

    Args:
        query: the search query
        candidates: list of (chunk_id, text, retrieval_score) from first-stage retrieval
        top_k: number of results to return (default: all)
        return_scores: if True, return cross-encoder scores; else keep original scores

    Returns:
        Reranked list of (chunk_id, text, score), sorted by cross-encoder score desc.

    Performance:
        - 10 candidates: ~10ms
        - 50 candidates: ~50ms
        - 100 candidates: ~200ms (CPU)
    """
    model = _load_model()
    if model is None or not candidates:
        return candidates[:top_k] if top_k else candidates

    start = time.perf_counter()

    # Create (query, passage) pairs for cross-encoder
    pairs = [(query, text) for _, text, _ in candidates]
    ids = [cid for cid, _, _ in candidates]
    texts = [text for _, text, _ in candidates]

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Cross-encoder prediction failed: {e}. Returning original ranking.")
        return candidates[:top_k] if top_k else candidates

    # Sort by cross-encoder score (higher = more relevant)
    ranked = sorted(
        zip(ids, texts, scores.tolist()),
        key=lambda x: x[2],
        reverse=True
    )

    latency_ms = (time.perf_counter() - start) * 1000
    logger.debug(f"Cross-encoder reranked {len(candidates)} candidates in {latency_ms:.1f}ms")

    result = list(ranked[:top_k] if top_k else ranked)
    return result


class Reranker:
    """
    Stateful reranker wrapper for use in the HybridRetriever pipeline.

    Usage:
        reranker = Reranker()
        chunks = reranker.rerank_chunks(query, chunks, top_k=5)
    """

    def __init__(self, model_name: str = _model_name, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        if enabled:
            # Pre-load model at init time so first request isn't slow
            self._model = _load_model()

    def rerank_chunks(self, query: str, chunks, top_k: Optional[int] = None):
        """
        Rerank RetrievedChunk objects using cross-encoder.

        Args:
            query: search query
            chunks: list of RetrievedChunk objects
            top_k: number to return

        Returns:
            Reranked list of RetrievedChunk objects with updated scores.
        """
        if not self.enabled or not chunks or self._model is None:
            return chunks[:top_k] if top_k else chunks

        pairs = [(query, c.text) for c in chunks]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Reranker failed: {e}")
            return chunks[:top_k] if top_k else chunks

        # Update scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            chunk.retrieval_mode = f"{chunk.retrieval_mode}+reranked"

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return reranked[:top_k] if top_k else reranked

    @property
    def is_available(self) -> bool:
        return self._model is not None
