"""
Ingestion Pipeline
- Chunking with overlap
- Sentence-transformer embeddings (all-MiniLM-L6-v2)
- Schema versioning
- Batch processing
"""

import hashlib
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    chunk_index: int
    embedding: np.ndarray = None
    metadata: Dict[str, Any] = None
    schema_version: str = settings.SCHEMA_VERSION


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Chunk text with sliding window overlap.
    Tries to split on sentence boundaries first.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    # Split on sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if current_len + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = overlap_words + sentence.split()
            current_len = len(current_chunk)
        else:
            current_chunk.extend(sentence.split())
            current_len += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID based on content."""
    content = f"{doc_id}:{chunk_index}:{text[:50]}"
    return hashlib.md5(content.encode()).hexdigest()


class EmbeddingPipeline:
    """
    Generates sentence-transformer embeddings.
    Model: all-MiniLM-L6-v2 (384-dim, fast, strong semantic quality)
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dim = settings.EMBEDDING_DIM

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of texts.
        Returns float32 numpy array of shape (N, dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class IngestionPipeline:
    """
    End-to-end ingestion:
    1. Chunk documents
    2. Generate embeddings
    3. Return chunks ready for indexing
    """

    def __init__(self, embedding_pipeline: EmbeddingPipeline = None):
        self.embedder = embedding_pipeline or EmbeddingPipeline()

    def process(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> Tuple[List[Chunk], int, int]:
        """
        Process documents into embedded chunks.

        Returns:
            chunks: list of Chunk objects with embeddings
            ingested: count of successfully processed docs
            failed: count of failed docs
        """
        all_chunks = []
        ingested = 0
        failed = 0

        for doc in documents:
            try:
                doc_id = doc.get("id", make_chunk_id("doc", 0, doc.get("text", "")))
                text = doc.get("text", "").strip()
                metadata = doc.get("metadata", {})
                schema_version = doc.get("schema_version", settings.SCHEMA_VERSION)

                if not text:
                    logger.warning(f"Empty text for doc {doc_id}, skipping")
                    failed += 1
                    continue

                # Chunk
                text_chunks = chunk_text(text)

                for i, chunk_text_str in enumerate(text_chunks):
                    chunk_id = make_chunk_id(doc_id, i, chunk_text_str)
                    chunk = Chunk(
                        id=chunk_id,
                        text=chunk_text_str,
                        doc_id=doc_id,
                        chunk_index=i,
                        metadata={**metadata, "doc_id": doc_id, "chunk_index": i},
                        schema_version=schema_version
                    )
                    all_chunks.append(chunk)

                ingested += 1

            except Exception as e:
                logger.error(f"Failed to process document: {e}")
                failed += 1

        # Batch embed all chunks
        if all_chunks:
            logger.info(f"Embedding {len(all_chunks)} chunks...")
            texts = [c.text for c in all_chunks]
            embeddings = self.embedder.embed(texts, batch_size=batch_size, show_progress=True)
            for chunk, emb in zip(all_chunks, embeddings):
                chunk.embedding = emb

        logger.info(f"Ingestion complete: {ingested} docs → {len(all_chunks)} chunks, {failed} failed")
        return all_chunks, ingested, failed
