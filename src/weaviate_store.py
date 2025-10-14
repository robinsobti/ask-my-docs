from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, TYPE_CHECKING

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.collections import collection
from weaviate.exceptions import WeaviateBaseError

from .config import COLLECTION_NAME, DOCS_SCHEMA, WEAVIATE_URL

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from numpy.typing import NDArray

    from .chunking import Chunk


def get_client() -> weaviate.WeaviateClient:
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(WEAVIATE_URL.split(":")[-1])
    return weaviate.connect_to_local(host=host, port=port)


def create_collection_if_missing() -> collection:
    client = get_client()
    try:
        existing_collections = list(client.collections.list_all())
        if COLLECTION_NAME not in existing_collections:
            props = [
                Property(
                    name=p["name"],
                    data_type=DataType.TEXT,
                    description=p.get("description"),
                )
                for p in DOCS_SCHEMA["properties"]
            ]
            client.collections.create(
                name=DOCS_SCHEMA["name"],
                description=DOCS_SCHEMA["description"],
                properties=props,
                vectorizer_config=Configure.Vectorizer.none(),
                inverted_index_config=Configure.inverted_index(),
            )
        return client.collections.get(COLLECTION_NAME)
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


def upsert_batch(objs: List[Dict[str, Any]]) -> int:
    """
    Insert documents (BM25-only). If an ID exists, we overwrite it.
    Each obj must include: id (uuid string), text, title, source, doc_id, chunk_id
    """
    coll = _resolve_collection(COLLECTION_NAME)
    count = 0
    for obj in objs:
        for field in ("id", "text", "title", "source", "doc_id", "chunk_id"):
            if field not in obj:
                raise ValueError(f"Missing field '{field}' in object: {obj}")
        try:
            coll.data.delete_by_id(obj["id"])
        except Exception:
            pass
        coll.data.insert(
            properties={
                "text": obj["text"],
                "title": obj["title"],
                "source": obj["source"],
                "doc_id": obj["doc_id"],
                "chunk_id": obj["chunk_id"],
            },
            uuid=obj["id"],
        )
        count += 1
    return count


def upsert_batch_with_vectors(
    collection_name: str,
    chunks: List["Chunk"],
    vectors: "NDArray[float] | Sequence[Sequence[float]]",
) -> int:
    """
    Insert chunk objects into `collection_name` using explicit vectors.
    The first vector written determines the collection's vector dimension: ensure it matches your embedder.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk count ({len(chunks)}) and vector count ({len(vectors)}) must match."
        )
    if not chunks:
        return 0

    vector_rows = _normalize_vectors(vectors)
    coll = _resolve_collection(collection_name)

    count = 0
    for chunk, vector in zip(chunks, vector_rows):
        metadata = chunk.metadata or {}
        try:
            coll.data.delete_by_id(chunk.id)
        except Exception:
            pass
        coll.data.insert(
            properties={
                "text": chunk.text,
                "title": chunk.title,
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.id,
            },
            uuid=chunk.id,
            vector=vector,
        )
        count += 1
    return count


def search_bm25(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    BM25 keyword search (no vectors).
    Returns list of dicts with score, title, text, source, id.
    """
    coll = _resolve_collection(COLLECTION_NAME)
    res = coll.query.bm25(query=query, limit=k)
    hits = []
    for obj in res.objects:
        hits.append(
            {
                "id": obj.uuid,
                "score": obj.metadata.score,
                "title": obj.properties.get("title", ""),
                "text": obj.properties.get("text", ""),
                "source": obj.properties.get("source", ""),
            }
        )
    return hits
