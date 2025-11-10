#!/usr/bin/env python3
"""
CLI for ingesting the curated dental PDFs into the configured vector store.

Usage:
    python scripts/ingest.py  # defaults to data/dental_corpus
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np

# Ensure project root is on sys.path so `from src...` imports work when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "dental_corpus"

from src.chunking import load_files, split_into_chunks  # noqa: E402
from src.config import COLLECTION_NAME, DOCS_SCHEMA  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.vector_store import close_client, create_collection_if_missing, upsert_batch  # noqa: E402

logger = logging.getLogger(__name__)


def _iter_docs(paths: Iterable[str]) -> Iterator[dict]:
    for raw_doc in load_files(list(paths)):
        doc = dict(raw_doc)
        metadata_payload = dict(doc.pop("metadata", {}) or {})
        filename = metadata_payload.get("filename") or doc.get("title") or doc.get("doc_id") or ""
        book_name = metadata_payload.get("book_name") or Path(filename).stem or filename
        doc["metadata"] = {
            "book_name": book_name,
            "filename": filename,
        }
        doc["title"] = book_name or doc.get("title") or doc.get("doc_id") or ""
        doc["source"] = filename
        yield doc


def _build_objects(chunks: List[dict], vectors: np.ndarray) -> List[dict]:
    objects: List[dict] = []
    for idx, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", "")
        chunk_id = chunk.get("chunk_id", "")
        uid = f"{doc_id}::{chunk_id}"
        metadata = chunk.get("metadata") or {}
        objects.append(
            {
                "id": uid,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "title": chunk.get("title", ""),
                "book_name": metadata.get("book_name") or chunk.get("title", ""),
                "source": metadata.get("filename") or chunk.get("source", ""),
                "text": chunk.get("text", ""),
                "ord": chunk.get("ord", 0),
                "metadata": metadata,
                "vector": vectors[idx],
            }
        )
    return objects


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into the configured vector store.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=str(DEFAULT_DATA_DIR),
        help=f"Path to the directory containing documents to ingest (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters (default: 800).")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters (default: 120).")
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI text-embedding-3-small model to use for embeddings.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Vector store collection/index name (default: {COLLECTION_NAME}).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_path}' does not exist.")

    logger.info("Scanning %s for documents", data_path)

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
    batch_objects: List[dict] = []

    doc_iter = _iter_docs([str(data_path)])

    for doc in doc_iter:
        total_docs += 1
        text_length = len(doc.get("text", ""))
        logger.info("Chunking doc=%s (%d chars)", doc.get("doc_id"), text_length)
        chunks = list(
            split_into_chunks(
                doc,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
        )
        if not chunks:
            logger.warning("No chunks produced for doc=%s", doc.get("doc_id"))
            continue

        metadata = doc.get("metadata", {}) or {}
        book_name = metadata.get("book_name") or doc.get("title") or doc.get("doc_id")
        for chunk in chunks:
            chunk.setdefault("metadata", {})
            chunk["metadata"].setdefault("book_name", book_name)
            chunk["metadata"].setdefault("filename", metadata.get("filename"))
            chunk["title"] = doc.get("title", book_name)
            chunk["source"] = metadata.get("filename") or chunk.get("source")

        texts = [chunk["text"] for chunk in chunks]
        vectors = embedder.encode(texts, normalize=True)
        objects = _build_objects(chunks, vectors)

        total_chunks += len(objects)
        batch_objects.extend(objects)

    if total_docs == 0:
        logger.warning("No documents found under %s", data_path)

    upserted = upsert_batch(batch_objects, collection_name=args.collection)

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
