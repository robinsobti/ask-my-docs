"""Vector store dispatch layer.

Selects the configured backend at import time and re-exports the common
interface expected by callers (retriever, ingest scripts, etc.).
"""

from __future__ import annotations

from src.config import VECTOR_STORE_PROVIDER

_SUPPORTED_PROVIDERS = {"weaviate", "pinecone"}

if VECTOR_STORE_PROVIDER not in _SUPPORTED_PROVIDERS:
    raise ValueError(f"Unsupported vector store provider '{VECTOR_STORE_PROVIDER}'.")

if VECTOR_STORE_PROVIDER == "pinecone":
    from . import pinecone_store as _backend
else:
    from . import weaviate_store as _backend


create_collection_if_missing = _backend.create_collection_if_missing
upsert_batch = _backend.upsert_batch
search_bm25 = _backend.search_bm25
search_vector = _backend.search_vector
close_client = _backend.close_client
