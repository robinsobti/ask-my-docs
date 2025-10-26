"""
Retrieval utilities supporting BM25, vector, and hybrid querying against the
configured vector store backing this project.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TYPE_CHECKING, cast

import numpy as np
from src.embedder import Embedder

from .config import (
    COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_TOP_K,
    VECTOR_STORE_PROVIDER,
)
from .vector_store import create_collection_if_missing, search_bm25, search_vector

# if TYPE_CHECKING:  # pragma: no cover - import for typing only
#     from .embedder import Embedder

Mode = Literal["bm25", "vector", "hybrid"]


def _normalize_mode(mode: str) -> Mode:
    normalized = mode.lower()
    if normalized not in {"bm25", "vector", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Expected bm25, vector, or hybrid.")
    return cast(Mode, normalized)


def _require_embedder(embedder: Optional["Embedder"], mode: Mode) -> "Embedder":
    if embedder is None:
        raise ValueError(f"Retrieval mode '{mode}' requires an embedder instance.")
    return embedder


def retrieve(
    query: str,
    mode: str = DEFAULT_RETRIEVAL_MODE,
    k: int = DEFAULT_RETRIEVAL_TOP_K,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    *,
    collection: str = COLLECTION_NAME,
    embedder: Optional["Embedder"] = None,
) -> List[Dict[str, object]]:
    """
    Retrieve documents using BM25, vector, or hybrid scoring.
    Returns a list of hits: { "id", "score", "text", "title", "source" }.
    """
    mode_normalized = _normalize_mode(mode)
    if k <= 0:
        raise ValueError("k must be > 0.")

    query = query.strip()
    if not query:
        return []

    if mode_normalized == "hybrid" and not 0.0 <= alpha <= 1.0:
        raise ValueError("Hybrid alpha must be between 0.0 and 1.0.")

    # Ensure the collection exists before querying.
    create_collection_if_missing(name=collection)

    if mode_normalized == "bm25":
        if VECTOR_STORE_PROVIDER != "weaviate":
            raise NotImplementedError("BM25 mode is only available with the Weaviate provider.")
        return search_bm25(query=query, k=k, collection_name=collection)

    embed = _require_embedder(embedder, mode_normalized)
    query_vector = np.asarray(embed.encode(texts=[query], normalize=True)[0])

    if mode_normalized == "vector":
        return search_vector(query_vector, k=k, collection_name=collection)

    # Hybrid: fuse BM25 + vector scores.
    if VECTOR_STORE_PROVIDER != "weaviate":
        raise NotImplementedError("Hybrid retrieval requires BM25 support; switch to Weaviate or use vector mode.")

    bm25_hits = search_bm25(query=query, k=k, collection_name=collection)
    vector_hits = search_vector(query_vector, k=k, collection_name=collection)
    fused = fuse_hybrid(bm25_hits, vector_hits, alpha=alpha)
    return fused[:k]

def retrieve_with_vector(
    q_vec: np.ndarray,
    k: int = DEFAULT_RETRIEVAL_TOP_K,
    *,
    collection: str = COLLECTION_NAME,
) -> List[Dict[str, object]]:
    """
    Convenience helper when the caller already has a normalized query vector.
    """
    if k <= 0:
        raise ValueError("k must be > 0.")
    if q_vec.ndim != 1:
        raise ValueError("q_vec must be a 1-D numpy array.")
    create_collection_if_missing(name=collection)
    return search_vector(q_vec, k=k, collection_name=collection)


def fuse_hybrid(
    bm25_hits: List[Dict[str, object]],
    vector_hits: List[Dict[str, object]],
    alpha: float = DEFAULT_HYBRID_ALPHA,
) -> List[Dict[str, object]]:
    """
    Combine lexical and vector scores by weighted sum (score-level fusion).
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Hybrid alpha must be between 0.0 and 1.0.")

    combined: Dict[str, Dict[str, object]] = {}

    for hit in bm25_hits:
        combined[hit["id"]] = {
            "id": hit["id"],
            "title": hit.get("title", ""),
            "text": hit.get("text", ""),
            "source": hit.get("source", ""),
            "bm25_score": float(hit.get("score", 0.0) or 0.0),
            "vector_score": 0.0,
        }

    for hit in vector_hits:
        entry = combined.setdefault(
            hit["id"],
            {
                "id": hit["id"],
                "title": hit.get("title", ""),
                "text": hit.get("text", ""),
                "source": hit.get("source", ""),
                "bm25_score": 0.0,
                "vector_score": 0.0,
            },
        )
        entry["vector_score"] = float(hit.get("score", 0.0) or 0.0)
        # Prefer non-empty metadata from vector result
        for key in ("title", "text", "source"):
            if not entry.get(key):
                entry[key] = hit.get(key, "")

    fused: List[Dict[str, object]] = []
    for entry in combined.values():
        bm_score = float(entry.get("bm25_score", 0.0) or 0.0)
        vec_score = float(entry.get("vector_score", 0.0) or 0.0)
        fused.append(
            {
                "id": entry["id"],
                "title": entry.get("title", ""),
                "text": entry.get("text", ""),
                "source": entry.get("source", ""),
                "score": alpha * bm_score + (1.0 - alpha) * vec_score,
            }
        )

    fused.sort(key=lambda hit: hit["score"], reverse=True)
    return fused
