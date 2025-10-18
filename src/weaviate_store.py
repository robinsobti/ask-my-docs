from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Optional
import uuid

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections import collection
from weaviate.classes.query import MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateBaseError

from .config import (
    COLLECTION_NAME,
    DEFAULT_UPSERT_BATCH_SIZE,
    DOCS_SCHEMA,
    WEAVIATE_URL,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from numpy.typing import NDArray

    from .chunking import Chunk


_CLIENT: Optional[weaviate.WeaviateClient] = None


def get_client() -> weaviate.WeaviateClient:
    global _CLIENT
    if _CLIENT is None:
        host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
        port = int(WEAVIATE_URL.split(":")[-1])
        _CLIENT = weaviate.connect_to_local(host=host, port=port)
    return _CLIENT


def close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        _CLIENT.close()
        _CLIENT = None


def create_collection_if_missing(
    name: str = COLLECTION_NAME,
    properties: List[Tuple[str, str, str | None]] | None = None,
    bm25_enabled: bool = True,
    vector_dim: int | None = None,
) -> collection:
    client = get_client()
    target_name = name or COLLECTION_NAME
    try:
        coll = client.collections.get(target_name)
        if not coll.exists():
            if properties is None:
                properties = [
                    (
                        prop["name"],
                        prop.get("dataType", "text"),
                        prop.get("description", ""),
                    )
                    for prop in DOCS_SCHEMA["properties"]
                ]

            props: List[Property] = []
            for entry in properties:
                if len(entry) == 2:
                    prop_name, prop_type = entry
                    prop_desc = ""
                else:
                    prop_name, prop_type, prop_desc = entry

                dtype = (prop_type or "text").lower()
                if dtype == "int":
                    data_type = DataType.INT
                elif dtype in {"number", "float"}:
                    data_type = DataType.NUMBER
                else:
                    data_type = DataType.TEXT

                props.append(
                    Property(
                        name=prop_name,
                        data_type=data_type,
                        description=prop_desc or None,
                    )
                )
            inverted_conf = Configure.inverted_index()
            vector_index_conf = Configure.VectorIndex.hnsw()
            vector_config = Configure.Vectors.self_provided(
                name="default",
                vector_index_config=vector_index_conf,
            )
            client.collections.create(
                name=target_name,
                description=DOCS_SCHEMA["description"],
                properties=props,
                vector_config=vector_config,
                inverted_index_config=inverted_conf,
            )
            coll = client.collections.get(target_name)
        return coll
    except WeaviateBaseError as exc:
        raise RuntimeError(f"Error creating collection: {exc}") from exc


def _resolve_collection(collection_name: str) -> collection:
    if collection_name == COLLECTION_NAME:
        return create_collection_if_missing()
    client = get_client()
    return client.collections.get(collection_name)


def _normalize_vectors(
    vectors: "NDArray[float] | Sequence[Sequence[float]]",
) -> List[List[float]]:
    """
    Convert numpy arrays or nested sequences into a validated list of equal-length float vectors.
    """
    raw_vectors: Any
    if hasattr(vectors, "tolist"):
        raw_vectors = vectors.tolist()
    else:
        raw_vectors = vectors

    if not isinstance(raw_vectors, Sequence):
        raw_vectors = list(raw_vectors)

    normalized: List[List[float]] = []
    expected_dim: int | None = None
    for index, vector in enumerate(raw_vectors):
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, Sequence):
            raise TypeError(f"Vector at position {index} is not a sequence.")
        vector_list = [float(value) for value in vector]
        if expected_dim is None:
            expected_dim = len(vector_list)
            if expected_dim == 0:
                raise ValueError("Vector dimension cannot be zero.")
        elif len(vector_list) != expected_dim:
            raise ValueError(
                f"Vector at position {index} has dimension {len(vector_list)}; expected {expected_dim}."
            )
        normalized.append(vector_list)

    return normalized


def upsert_batch(
    objects: Iterable[Dict[str, Any]],
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Insert or update chunk objects (properties + vectors) into the target collection.
    Each object must include: id, text, title, source, doc_id, chunk_id, ord, vector.
    `batch_size` is retained for API compatibility but is ignored in the v2 ingestion flow.
    """
    coll = _resolve_collection(collection_name)

    def _normalize_object(obj: Dict[str, Any]) -> Tuple[uuid.UUID, Dict[str, Any], List[float]]:
        missing = [
            field
            for field in ("id", "text", "title", "source", "doc_id", "chunk_id", "ord", "vector")
            if field not in obj
        ]
        if missing:
            raise ValueError(f"Missing required fields {missing} in object: {obj}")

        stable_id = str(obj["id"])
        uid = uuid.uuid5(uuid.NAMESPACE_URL, stable_id)
        vector = obj["vector"]
        vector_row = _normalize_vectors([vector])[0]

        properties = {
            "id": stable_id,
            "stable_id": stable_id,
            "text": str(obj["text"] or ""),
            "title": str(obj.get("title") or ""),
            "source": str(obj.get("source") or ""),
            "doc_id": str(obj.get("doc_id") or ""),
            "chunk_id": str(obj.get("chunk_id") or ""),
            "ord": int(obj.get("ord", 0) or 0),
        }
        return uid, properties, vector_row

    total = 0
    for raw_obj in objects:
        uid, props, vector_row = _normalize_object(raw_obj)
        try:
            coll.data.insert(properties=props, uuid=uid, vector=vector_row)
        except UnexpectedStatusCodeError as exc:
            status = getattr(exc, "status_code", None)
            message = getattr(exc, "message", None) or str(exc)
            duplicate = status in {409, 422} or "already exists" in message.lower()
            if not duplicate:
                raise
            coll.data.replace(uuid=uid, properties=props, vector=vector_row)
        total += 1

    return total

def _format_objects(objects: Sequence[Any]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for obj in objects:
        properties = getattr(obj, "properties", {}) or {}
        metadata = getattr(obj, "metadata", None)
        score = getattr(metadata, "score", None)
        if score is None:
            certainty = getattr(metadata, "certainty", None)
            distance = getattr(metadata, "distance", None)
            if certainty is not None:
                score = float(certainty)
            elif distance is not None:
                try:
                    score = max(0.0, 1.0 - float(distance))
                except (TypeError, ValueError):
                    score = None
        stable_id = properties.get("id") or properties.get("stable_id")
        if not stable_id:
            doc_part = properties.get("doc_id", "")
            chunk_part = properties.get("chunk_id", "")
            if doc_part or chunk_part:
                stable_id = f"{doc_part}::{chunk_part}".strip(":")
            else:
                stable_id = getattr(obj, "uuid", "")

        hits.append(
            {
                "id": stable_id,
                "score": float(score) if score is not None else 0.0,
                "title": properties.get("title", ""),
                "text": properties.get("text", ""),
                "source": properties.get("source", ""),
            }
        )
    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits


def search_bm25(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    BM25 keyword search (no vectors).
    Returns list of dicts with score, title, text, source, id.
    """
    coll = _resolve_collection(collection_name)
    res = coll.query.bm25(query=query, limit=k)
    hits = _format_objects(getattr(res, "objects", []))

    if not hits:
        return hits

    scores = [hit["score"] for hit in hits]
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        for hit in hits:
            hit["score"] = 1.0
        return hits

    span = max_score - min_score
    for hit in hits:
        hit["score"] = max(0.0, (hit["score"] - min_score) / span)

    hits.sort(key=lambda item: item["score"], reverse=True)
    return hits


def search_vector(
    q_vec: np.ndarray,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> List[Dict[str, Any]]:
    """
    Vector (semantic) search using a pre-computed query vector.
    Returns normalized hits with score derived from vector distance.
    """
    if q_vec.ndim != 1:
        raise ValueError("q_vec must be a 1-D numpy array.")
    coll = _resolve_collection(collection_name)
    res = coll.query.near_vector(
        near_vector=q_vec.tolist(),
        limit=k,
        return_metadata=MetadataQuery(distance=True, certainty=True),
    )
    return _format_objects(getattr(res, "objects", []))
