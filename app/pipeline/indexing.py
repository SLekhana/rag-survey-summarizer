"""
Indexing Layer
- FAISS IVF index for fast approximate nearest neighbor search
- ChromaDB for persistent vector storage with metadata
- Schema versioning for index management
"""

import os
import logging
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.pipeline.ingestion import Chunk

logger = logging.getLogger(__name__)


class FAISSIndex:
    """
    FAISS IVF (Inverted File Index) for scalable ANN search.
    - IVF with flat quantizer: good balance of speed and accuracy
    - L2-normalized vectors → inner product = cosine similarity
    """

    def __init__(self, dim: int = None, nlist: int = None):
        self.dim = dim or settings.EMBEDDING_DIM
        self.nlist = nlist or settings.FAISS_NLIST
        self.nprobe = settings.FAISS_NPROBE
        self.index = None
        self.id_map: Dict[int, str] = {}      # faiss int id → chunk id
        self.chunk_map: Dict[str, str] = {}    # chunk id → text
        self._next_id = 0

    def build(self, embeddings: np.ndarray, chunk_ids: List[str], texts: List[str]):
        """Build IVF index from scratch."""
        n = len(embeddings)
        logger.info(f"Building FAISS IVF index with {n} vectors, nlist={self.nlist}")

        # IVF flat index (exact distance within cells)
        quantizer = faiss.IndexFlatIP(self.dim)  # Inner product (cosine with normalized vecs)
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, min(self.nlist, n))
        self.index.nprobe = self.nprobe

        # Train on data
        self.index.train(embeddings)
        self.index.add(embeddings)

        # Build maps
        for i, (cid, text) in enumerate(zip(chunk_ids, texts)):
            self.id_map[i] = cid
            self.chunk_map[cid] = text

        self._next_id = n
        logger.info(f"FAISS index built: {self.index.ntotal} vectors")

    def add(self, embeddings: np.ndarray, chunk_ids: List[str], texts: List[str]):
        """Add new vectors to existing index."""
        if self.index is None:
            self.build(embeddings, chunk_ids, texts)
            return

        start_id = self._next_id
        self.index.add(embeddings)
        for i, (cid, text) in enumerate(zip(chunk_ids, texts)):
            self.id_map[start_id + i] = cid
            self.chunk_map[cid] = text
        self._next_id += len(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors.
        Returns list of (chunk_id, score) sorted by score desc.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.id_map:
                results.append((self.id_map[idx], float(score)))

        return results

    def save(self, path: str = None):
        path = path or settings.FAISS_INDEX_PATH
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/maps.pkl", "wb") as f:
            pickle.dump({"id_map": self.id_map, "chunk_map": self.chunk_map, "next_id": self._next_id}, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str = None):
        path = path or settings.FAISS_INDEX_PATH
        index_file = f"{path}/index.faiss"
        maps_file = f"{path}/maps.pkl"
        if not os.path.exists(index_file):
            logger.warning(f"No FAISS index found at {path}")
            return False
        self.index = faiss.read_index(index_file)
        self.index.nprobe = self.nprobe
        with open(maps_file, "rb") as f:
            data = pickle.load(f)
        self.id_map = data["id_map"]
        self.chunk_map = data["chunk_map"]
        self._next_id = data["next_id"]
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        return True

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0


class ChromaStore:
    """
    ChromaDB persistent vector store.
    Used for metadata filtering, persistent storage, and schema versioning.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection '{settings.CHROMA_COLLECTION}' ready: {self.collection.count()} docs")

    def add(self, chunks: List[Chunk]):
        """Add chunks with embeddings to ChromaDB."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        embeddings = [c.embedding.tolist() for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{
            **(c.metadata or {}),
            "schema_version": c.schema_version,
            "chunk_index": c.chunk_index,
            "doc_id": c.doc_id
        } for c in chunks]

        # Upsert in batches of 500 (ChromaDB limit)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        logger.info(f"Added {len(chunks)} chunks to ChromaDB")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: Optional[Dict] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Query ChromaDB.
        Returns list of (chunk_id, text, distance).
        """
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(top_k, max(1, self.collection.count())),
            "include": ["documents", "distances", "metadatas"]
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        if results["ids"] and results["ids"][0]:
            for cid, doc, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0]
            ):
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score 0-1
                score = 1 - (dist / 2)
                output.append((cid, doc, score))

        return output

    @property
    def total(self) -> int:
        return self.collection.count()

    def get_schema_versions(self) -> Dict[str, int]:
        """Count documents per schema version."""
        results = self.collection.get(include=["metadatas"])
        versions = {}
        for meta in results["metadatas"]:
            v = meta.get("schema_version", "unknown")
            versions[v] = versions.get(v, 0) + 1
        return versions
