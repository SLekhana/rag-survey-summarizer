from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = "your-openai-api-key"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_MODEL_STRONG: str = "gpt-4"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # FAISS
    FAISS_INDEX_PATH: str = "data/faiss_index"
    FAISS_NLIST: int = 50         # IVF clusters
    FAISS_NPROBE: int = 10        # clusters to search

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION: str = "survey_responses"

    # PostgreSQL
    POSTGRES_URL: str = "postgresql://user:password@localhost:5432/rag_db"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    TOP_K: int = 10
    BM25_WEIGHT: float = 0.4
    DENSE_WEIGHT: float = 0.6

    # Schema versioning
    SCHEMA_VERSION: str = "1.0.0"

    # App
    APP_TITLE: str = "RAG Survey Response Summarizer"
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
