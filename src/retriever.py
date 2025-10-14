"""
Retrieval utilities supporting BM25, vector, and hybrid querying against the
Weaviate collection defined in this project.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TYPE_CHECKING, cast

from .config import COLLECTION_NAME
from .weaviate_store import create_collection_if_missing, get_client

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .embedder import Embedder

Mode = Literal["bm25", "vector", "hybrid"]

_PROPERTIES = ["title", "text", "source"]


def _normalize_mode(mode: str) -> Mode:
    normalized = mode.lower()
    if normalized not in {"bm25", "vector", "hybrid"}:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Expected bm25, vector, or hybrid.")
    return cast(Mode, normalized)


def _require_embedder(embedder: Optional["Embedder"], mode: Mode) -> "Embedder":
    if embedder is None:
        raise ValueError(f"Retrieval mode '{mode}' requires an embedder instance.")
    return embedder


def _extract_score(metadata: object) -> float:
    """
    Attempt to derive a comparable score from Weaviate metadata objects.
    Works for BM25 (score), hybrid (score), and vector (distance/certainty).
    """
    if metadata is None:
        return 0.0

    score = getattr(metadata, "score", None)
    if score is not None:
        return float(score)

    certainty = getattr(metadata, "certainty", None)
    if certainty is not None:
        return float(certainty)

    distance = getattr(metadata, "distance", None)
    if distance is not None:
        try:
            return float(1.0 - float(distance))
        except (TypeError, ValueError):
            pass

    return 0.0


def _format_hits(objects: List[object]) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    for obj in objects:
        properties = getattr(obj, "properties", {}) or {}
        hits.append(
            {
                "id": getattr(obj, "uuid", ""),
                "score": _extract_score(getattr(obj, "metadata", None)),
                "text": properties.get("text", ""),
                "title": properties.get("title", ""),
                "source": properties.get("source", ""),
            }
        )

    # Ensure descending order even if backend already sorts.
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


def retrieve(
    collection: str,
    query: str,
    k: int = 5,
    mode: str = "hybrid",
    alpha: float = 0.5,
    embedder: Optional["Embedder"] = None,
) -> List[Dict[str, object]]:
    """
    Retrieve documents using BM25, vector, or hybrid scoring.
    Returns a list of hits: { "id", "score", "text", "title", "source" }.
    """
    mode_normalized = _normalize_mode(mode)

    if k <= 0:
        return []

    if mode_normalized == "hybrid" and not 0.0 <= alpha <= 1.0:
        raise ValueError("Hybrid alpha must be between 0.0 and 1.0.")

    if collection == COLLECTION_NAME:
        coll = create_collection_if_missing()
    else:
        client = get_client()
        coll = client.collections.get(collection)

    if mode_normalized == "bm25":
        result = coll.query.bm25(query=query, limit=k, return_properties=_PROPERTIES)
        return _format_hits(getattr(result, "objects", []))

    query_vector = _require_embedder(embedder, mode_normalized).encode([query])[0].tolist()

    if mode_normalized == "vector":
        result = coll.query.near_vector(vector=query_vector, limit=k, return_properties=_PROPERTIES)
        return _format_hits(getattr(result, "objects", []))

    # mode == "hybrid"
    result = coll.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,
        limit=k,
        return_properties=_PROPERTIES,
    )
    return _format_hits(getattr(result, "objects", []))
