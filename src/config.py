import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer. Got {value!r}.") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float. Got {value!r}.") from exc


def _env_float_optional(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float. Got {value!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean (got {value!r}).")


_ALLOWED_VECTOR_STORE_PROVIDERS = {"weaviate", "pinecone"}
VECTOR_STORE_PROVIDER = (os.getenv("VECTOR_STORE_PROVIDER", "weaviate") or "weaviate").strip().lower()
if VECTOR_STORE_PROVIDER not in _ALLOWED_VECTOR_STORE_PROVIDERS:
    VECTOR_STORE_PROVIDER = "weaviate"

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "ask-my-docs")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "ask-my-docs")

# Chunking defaults
DEFAULT_CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
DEFAULT_CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
CHUNK_BOUNDARY_WINDOW = _env_int("CHUNK_BOUNDARY_WINDOW", 50)

# Embedder defaults
DEFAULT_EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "text-embedding-3-small")
DEFAULT_EMBED_DIM = _env_int("DEFAULT_EMBED_DIM", 512)
DEFAULT_EMBEDDER_BATCH_SIZE = _env_int("EMBEDDER_BATCH_SIZE", 32)

# Generator defaults
DEFAULT_GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")

# Retrieval defaults
_ALLOWED_RETRIEVAL_MODES = {"bm25", "vector", "hybrid"}
DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
if DEFAULT_RETRIEVAL_MODE not in _ALLOWED_RETRIEVAL_MODES:
    DEFAULT_RETRIEVAL_MODE = "vector"
DEFAULT_RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 2)
DEFAULT_HYBRID_ALPHA = _env_float("RETRIEVAL_ALPHA", 0.5)
DEFAULT_RERANK_DEPTH = _env_int("RERANK_DEPTH", 25)
ENABLE_RERANK = _env_bool("ENABLE_RERANK", True)
RERANK_BACKEND = (os.getenv("RERANK_BACKEND", "hf") or "hf").strip().lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2") or "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 8)
RERANK_TIMEOUT_S = _env_float("RERANK_TIMEOUT_S", 15.0)

# OPEN AI API defaults
OPENAI_MAX_TOKENS = _env_int("OPENAI_MAX_TOKENS", 512)
OPENAI_TEMPERATURE = _env_float("OPENAI_TEMPERATURE", 0.2)
OPENAI_TOP_P = _env_float("OPENAI_TOP_P", 1.0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Pricing (optional; set via env)
PRICE_PROMPT_PER_1K = _env_float_optional("PRICE_PROMPT_PER_1K")
PRICE_COMPLETION_PER_1K = _env_float_optional("PRICE_COMPLETION_PER_1K")

# Upsert batching defaults
DEFAULT_UPSERT_BATCH_SIZE = _env_int("UPSERT_BATCH_SIZE", 100)

# Schema definition for the Docs collection used in v1 (BM25 + client vectors).
DOCS_SCHEMA = {
    "name": COLLECTION_NAME,
    "description": "Text chunks from documents (BM25 enabled; vectors stored client-side)",
    "properties": [
        {"name": "id", "dataType": "text", "description": "Stable chunk identifier doc_id::chunk_id"},
        {"name": "stable_id", "dataType": "text", "description": "Legacy chunk identifier (doc_id::chunk_id)"},
        {"name": "text", "dataType": "text", "description": "Chunk text"},
        {"name": "title", "dataType": "text", "description": "Document title"},
        {"name": "book_name", "dataType": "text", "description": "Name of the PDF/book the chunk came from"},
        {"name": "source", "dataType": "text", "description": "Source path/URL"},
        {"name": "doc_id", "dataType": "text", "description": "Stable document id"},
        {"name": "chunk_id", "dataType": "text", "description": "Stable chunk id"},
        {"name": "ord", "dataType": "int", "description": "Chunk order within the original document"},
    ],
    "vectorizer": "none",
    "moduleConfig": {
        "bm25": {"enabled": True}
    }
}
