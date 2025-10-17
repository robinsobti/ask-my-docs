#!/usr/bin/env python3
"""
CLI for ingesting a directory of documents into the local Weaviate instance.

Usage:
    python scripts/ingest.py data/raw_docs --chunk-size 800 --chunk-overlap 120 --model all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np

# Ensure project root is on sys.path so `from src...` imports work when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import load_files, split_into_chunks  # noqa: E402
from src.config import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDER_MODEL,
    DOCS_SCHEMA,
)
from src.embedder import Embedder  # noqa: E402
from src.weaviate_store import close_client, create_collection_if_missing, upsert_batch  # noqa: E402


def _iter_docs(paths: Iterable[str]) -> Iterable[dict]:
    return load_files(list(paths))


def _build_objects(chunks: List[dict], vectors: np.ndarray) -> List[dict]:
    objects: List[dict] = []
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", "")
        chunk_id = chunk.get("chunk_id", "")
        uid = f"{doc_id}::{chunk_id}"
        objects.append(
            {
                "id": uid,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": chunk.get("title", ""),
                "source": chunk.get("source", ""),
                "text": chunk.get("text", ""),
                "ord": chunk.get("ord", 0),
                "vector": vectors[idx],
            }
        )
    return objects

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Weaviate.")
    parser.add_argument("data_dir", help="Path to the directory containing documents to ingest.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDER_MODEL,
        help=f"Sentence-transformers model to use for embeddings (default: {DEFAULT_EMBEDDER_MODEL}).",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Weaviate collection name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")

    embedder = Embedder(model_name=args.model)

    create_collection_if_missing(
        name=args.collection,
        properties=[
            (prop["name"], prop.get("dataType", "text"), prop.get("description", ""))
            for prop in DOCS_SCHEMA["properties"]
        ],
        bm25_enabled=True,
        vector_dim=embedder.dim,
    )

    total_docs = 0
    total_chunks = 0
    upserted = 0

    for doc in _iter_docs([str(data_path)]):
        total_docs += 1
        chunks = list(
            split_into_chunks(
                doc,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
        if not chunks:
            continue

        texts = [chunk["text"] for chunk in chunks]
        vectors = embedder.encode(texts, normalize=True)
        objects = _build_objects(chunks, vectors)

        total_chunks += len(objects)
        upserted += upsert_batch(objects, collection_name=args.collection)

    print(
        f"Ingest complete: docs={total_docs}, chunks={total_chunks}, upserted={upserted}, "
        f"collection='{args.collection}', model='{args.model}'."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_client()
