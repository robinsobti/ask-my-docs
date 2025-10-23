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


WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION", "Docs")

# Chunking defaults
DEFAULT_CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
DEFAULT_CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
CHUNK_BOUNDARY_WINDOW = _env_int("CHUNK_BOUNDARY_WINDOW", 50)

# Embedder defaults
DEFAULT_EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_EMBEDDER_BATCH_SIZE = _env_int("EMBEDDER_BATCH_SIZE", 32)
DEFAULT_EMBEDDER_DEVICE: Optional[str] = os.getenv("EMBEDDER_DEVICE") or None

# Generator defaults
DEFAULT_GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")

# Retrieval defaults
_ALLOWED_RETRIEVAL_MODES = {"bm25", "vector", "hybrid"}
DEFAULT_RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
if DEFAULT_RETRIEVAL_MODE not in _ALLOWED_RETRIEVAL_MODES:
    DEFAULT_RETRIEVAL_MODE = "hybrid"
DEFAULT_RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5)
DEFAULT_HYBRID_ALPHA = _env_float("RETRIEVAL_ALPHA", 0.5)
DEFAULT_RERANK_DEPTH = _env_int("RERANK_DEPTH", 25)

# OPEN AI API defaults
OPENAI_MAX_TOKENS = _env_int("RETRIEVAL_TOP_K", 1000)
OPENAI_TEMPERATURE = _env_float("RETRIEVAL_TOP_K", 0.3)
OPENAI_TOP_P = _env_float("RETRIEVAL_TOP_K", 0.3)
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
