#!/usr/bin/env python3
"""
CLI for querying the local Weaviate collection using BM25, vector, or hybrid retrieval.

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

from src.config import COLLECTION_NAME  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.retriever import Mode, retrieve  # noqa: E402
from src.weaviate_store import close_client  # noqa: E402


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the Weaviate collection.")
    parser.add_argument("--q", required=True, help="Query text.")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid", help="Retrieval mode.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid score weighting (only for hybrid mode).")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model to use for embeddings (vector/hybrid modes).",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Weaviate collection name (default: {COLLECTION_NAME}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

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
