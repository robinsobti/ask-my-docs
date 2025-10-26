#!/usr/bin/env python3
"""
CLI for querying the configured vector store using BM25, vector, or hybrid retrieval.

Usage:
    python scripts/search.py --q "reset my password" --mode hybrid --alpha 0.5 --k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so `from src...` imports work when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_TOP_K,
    VECTOR_STORE_PROVIDER,
)  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.retriever import Mode, retrieve  # noqa: E402
from src.vector_store import close_client  # noqa: E402


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the configured vector store.")
    parser.add_argument(
        "--q",
        default="reset my password",
        help="Query text (default: 'reset my password').",
    )
    parser.add_argument(
        "--mode",
        choices=["bm25", "vector", "hybrid"],
        default=DEFAULT_RETRIEVAL_MODE,
        help=f"Retrieval mode (default: {DEFAULT_RETRIEVAL_MODE}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_RETRIEVAL_TOP_K}).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_HYBRID_ALPHA,
        help=f"Hybrid score weighting (default: {DEFAULT_HYBRID_ALPHA}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDER_MODEL,
        help=f"Sentence-transformers model to use for embeddings (vector/hybrid modes). Default: {DEFAULT_EMBEDDER_MODEL}.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Vector store collection/index name (default: {COLLECTION_NAME}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    if VECTOR_STORE_PROVIDER == "pinecone" and args.mode in {"bm25", "hybrid"}:
        print("Provider 'pinecone' supports vector search only; falling back to --mode vector.")
        args.mode = "vector"

    embedder: Optional[Embedder] = None
    if args.mode in ("vector", "hybrid"):
        embedder = Embedder(model_name=args.model)

    try:
        hits = retrieve(
            query=args.q,
            mode=args.mode,
            k=args.k,
            alpha=args.alpha,
            collection=args.collection,
            embedder=embedder,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not hits:
        print("No results.")
        return 0

    print(f"Top {min(len(hits), args.k)} results for query '{args.q}' (mode={args.mode}):")
    for idx, hit in enumerate(hits[: args.k], start=1):
        title = hit.get("title") or "<no title>"
        source = hit.get("source") or "<no source>"
        snippet = (hit.get("text") or "")[:160].replace("\n", " ")
        print(f"{idx:>2}. score={hit['score']:.3f} title={title} source={source}")
        if snippet:
            print(f"    {snippet}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        close_client()
