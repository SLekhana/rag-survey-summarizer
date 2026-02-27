"""
Hybrid Retrieval Engine
- BM25 sparse retrieval (rank_bm25)
- FAISS dense retrieval
- Reciprocal Rank Fusion for hybrid combination
- Controlled experiment mode: compare sparse vs dense vs hybrid
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.pipeline.indexing import FAISSIndex, ChromaStore
from app.pipeline.ingestion import EmbeddingPipeline

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    retrieval_mode: str
    sparse_rank: Optional[int] = None
    dense_rank: Optional[int] = None
    metadata: Dict = None


def reciprocal_rank_fusion(
    sparse_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    sparse_weight: float = None,
    dense_weight: float = None,
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) for combining sparse and dense results.
    RRF score = sum(weight / (k + rank))
    """
    sparse_weight = sparse_weight or settings.BM25_WEIGHT
    dense_weight = dense_weight or settings.DENSE_WEIGHT

    scores: Dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(sparse_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + sparse_weight / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(dense_results):
        scores[chunk_id] = scores.get(chunk_id, 0) + dense_weight / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Retriever:
    """BM25 sparse retrieval over chunk texts."""

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_ids: List[str] = []
        self.corpus_texts: List[str] = []

    def index(self, chunk_ids: List[str], texts: List[str]):
        """Build BM25 index from chunks."""
        self.corpus_ids = chunk_ids
        self.corpus_texts = texts
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(texts)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search BM25 index. Returns (chunk_id, score) sorted desc."""
        if self.bm25 is None or not self.corpus_ids:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.corpus_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        return results

    @property
    def total(self) -> int:
        return len(self.corpus_ids)


class HybridRetriever:
    """
    Hybrid BM25 + FAISS retrieval pipeline.

    Supports 3 modes for controlled experiments:
    - sparse:  BM25 only
    - dense:   FAISS only
    - hybrid:  RRF fusion (default, best performance)
    """

    def __init__(
        self,
        faiss_index: FAISSIndex,
        chroma_store: ChromaStore,
        embedder: EmbeddingPipeline
    ):
        self.faiss = faiss_index
        self.chroma = chroma_store
        self.embedder = embedder
        self.bm25 = BM25Retriever()

    def build_bm25(self, chunk_ids: List[str], texts: List[str]):
        """Build BM25 index from the same corpus as FAISS."""
        self.bm25.index(chunk_ids, texts)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        mode: str = "hybrid",
        filters: Dict = None
    ) -> Tuple[List[RetrievedChunk], float]:
        """
        Retrieve relevant chunks.

        Args:
            query: search query
            top_k: number of results
            mode: "sparse" | "dense" | "hybrid"
            filters: optional ChromaDB metadata filters

        Returns:
            (chunks, latency_ms)
        """
        top_k = top_k or settings.TOP_K
        start = time.perf_counter()

        query_embedding = self.embedder.embed_single(query)

        sparse_results = []
        dense_results = []
        chunk_text_map: Dict[str, str] = {}

        # --- Sparse (BM25) ---
        if mode in ("sparse", "hybrid"):
            sparse_results = self.bm25.search(query, top_k=top_k * 2)
            for cid, score in sparse_results:
                if cid in self.bm25.corpus_ids:
                    idx = self.bm25.corpus_ids.index(cid)
                    chunk_text_map[cid] = self.bm25.corpus_texts[idx]

        # --- Dense (FAISS) ---
        if mode in ("dense", "hybrid"):
            dense_results = self.faiss.search(query_embedding, top_k=top_k * 2)
            for cid, score in dense_results:
                if cid in self.faiss.chunk_map:
                    chunk_text_map[cid] = self.faiss.chunk_map[cid]

        # --- Combine ---
        if mode == "sparse":
            ranked = sparse_results[:top_k]
            retrieval_mode = "sparse"
        elif mode == "dense":
            ranked = dense_results[:top_k]
            retrieval_mode = "dense"
        else:
            ranked = reciprocal_rank_fusion(sparse_results, dense_results)[:top_k]
            retrieval_mode = "hybrid"

        # Build sparse/dense rank maps for explainability
        sparse_rank_map = {cid: i for i, (cid, _) in enumerate(sparse_results)}
        dense_rank_map = {cid: i for i, (cid, _) in enumerate(dense_results)}

        chunks = []
        for cid, score in ranked:
            text = chunk_text_map.get(cid, "")
            if not text:
                # Fallback to ChromaDB
                chroma_results = self.chroma.query(query_embedding, top_k=1,
                                                    where={"chunk_id": cid} if filters else None)
                if chroma_results:
                    text = chroma_results[0][1]

            chunks.append(RetrievedChunk(
                id=cid,
                text=text,
                score=score,
                retrieval_mode=retrieval_mode,
                sparse_rank=sparse_rank_map.get(cid),
                dense_rank=dense_rank_map.get(cid),
                metadata={}
            ))

        latency_ms = (time.perf_counter() - start) * 1000
        return chunks, latency_ms
