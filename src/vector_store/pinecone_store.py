"""Pinecone-backed vector store helpers (implementation guidance only)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.config import (
    DEFAULT_EMBED_DIM,
    DEFAULT_UPSERT_BATCH_SIZE,
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
)

# Module-level caches for Pinecone primitives.
_CLIENT: Any = None
_INDEX_CACHE: Dict[str, Any] = {}


def _get_client() -> Any:
    """Lazy-initialize and return a Pinecone client instance."""

    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY must be set before using the Pinecone store.")
    # if not PINECONE_ENVIRONMENT:
    #     raise RuntimeError(
    #         "PINECONE_ENVIRONMENT must be set (e.g., 'us-east1-gcp') before using the Pinecone store."
    #     )

    try:
        import pinecone
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("The 'pinecone-client' package is required. Install with 'pip install pinecone-client'.") from exc

    _CLIENT = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    return _CLIENT


def _get_index(name: str | None = None, *, vector_dim: int | None = None) -> Any:
    """Return an index handle, creating it if necessary."""

    client = _get_client()
    index_name = (name or PINECONE_INDEX_NAME).strip()
    if not index_name:
        raise RuntimeError("collection_name/index name must be provided for Pinecone operations.")

    if index_name in _INDEX_CACHE:
        return _INDEX_CACHE[index_name]

    dim = int(vector_dim or DEFAULT_EMBED_DIM)
    raw_indexes = client.list_indexes()
    if hasattr(raw_indexes, "names"):
        existing_indexes = set(raw_indexes.names())
    else:
        indexes_attr = getattr(raw_indexes, "indexes", raw_indexes)
        existing_indexes = {
            item.name
            for item in indexes_attr
            if hasattr(item, "name") and item.name
        }

    if index_name not in existing_indexes:
        env = (PINECONE_ENVIRONMENT or "").strip()
        cloud = None
        region = None
        if env:
            parts = [part for part in env.replace("/", "-").split("-") if part]
            if parts:
                if parts[-1].lower() in {"aws", "gcp", "azure"}:
                    cloud = parts[-1].lower()
                    region = "-".join(parts[:-1])
                elif parts[0].lower() in {"aws", "gcp", "azure"}:
                    cloud = parts[0].lower()
                    region = "-".join(parts[1:])
        cloud = cloud or "aws"
        region = region or (env or "us-east-1")

        try:
            from pinecone import ServerlessSpec
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "The 'pinecone-client' package is required. Install with 'pip install pinecone-client'."
            ) from exc

        spec = ServerlessSpec(cloud=cloud, region=region)
        client.create_index(name=index_name, dimension=dim, metric="cosine", spec=spec)

    index = client.Index(index_name)
    _INDEX_CACHE[index_name] = index
    return index


def create_collection_if_missing(
    name: str,
    properties: List[Tuple[str, str, str | None]] | None = None,
    bm25_enabled: bool = True,
    vector_dim: int | None = None,
) -> Any:
    """Provision the Pinecone index if necessary and return the handle."""

    target_name = name or PINECONE_INDEX_NAME
    return _get_index(target_name, vector_dim=vector_dim)


def upsert_batch(
    objects: Iterable[Dict[str, Any]],
    batch_size: int = 100,
    collection_name: str = "",
) -> int:
    """Insert or update vectors in Pinecone."""

    if batch_size is None or batch_size <= 0:
        batch_size = DEFAULT_UPSERT_BATCH_SIZE

    index = _get_index(collection_name)

    payload: List[Dict[str, Any]] = []
    total = 0
    for obj in objects:
        if "id" not in obj or "vector" not in obj:
            raise ValueError("Each object must include 'id' and 'vector' fields for Pinecone upserts.")

        vector = obj["vector"]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if len(vector) != DEFAULT_EMBED_DIM:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match expected dimension {DEFAULT_EMBED_DIM}."
            )

        metadata: Dict[str, Any] = {}
        for key in ("text", "title", "book_name", "source", "doc_id", "chunk_id", "ord"):
            if key in obj and obj[key] is not None:
                metadata[key] = obj[key]
        payload.append({"id": str(obj["id"]), "values": vector, "metadata": metadata})
        total += 1

        if len(payload) >= batch_size:
            index.upsert(vectors=payload)
            payload.clear()

    if payload:
        index.upsert(vectors=payload)

    return total


def search_bm25(
    query: str,
    k: int = 5,
    collection_name: str = "",
) -> List[Dict[str, Any]]:
    """BM25 search is not available in Pinecone."""

    raise NotImplementedError("Pinecone does not support BM25 search; use vector mode instead.")


def search_vector(
    q_vec: Any,
    k: int = 5,
    collection_name: str = "",
) -> List[Dict[str, Any]]:
    """Semantic vector search against Pinecone."""

    if k <= 0:
        raise ValueError("k must be > 0.")

    if hasattr(q_vec, "tolist"):
        q_vec_list = q_vec.tolist()
    else:
        q_vec_list = list(q_vec)

    if len(q_vec_list) != DEFAULT_EMBED_DIM:
        raise ValueError(
            f"Query vector dimension {len(q_vec_list)} does not match expected dimension {DEFAULT_EMBED_DIM}."
        )

    index = _get_index(collection_name)
    response = index.query(vector=q_vec_list, top_k=k, include_metadata=True)

    matches = getattr(response, "matches", None) or response.get("matches", [])
    hits: List[Dict[str, Any]] = []
    for match in matches:
        metadata = getattr(match, "metadata", None) or match.get("metadata", {})
        score = getattr(match, "score", None) if hasattr(match, "score") else match.get("score")
        hits.append(
            {
                "id": getattr(match, "id", None) or match.get("id") or "",
                "score": float(score) if score is not None else 0.0,
                "title": metadata.get("title", ""),
                "book_name": metadata.get("book_name", ""),
                "text": metadata.get("text", ""),
                "source": metadata.get("source", ""),
            }
        )

    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits[:k]


def close_client() -> None:
    """Reset cached Pinecone resources."""

    global _CLIENT
    client = _CLIENT
    if client is not None and hasattr(client, "deinit"):
        try:
            client.deinit()
        except Exception:  # pragma: no cover - best effort cleanup
            pass
    _CLIENT = None
    _INDEX_CACHE.clear()
