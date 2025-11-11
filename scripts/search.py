#!/usr/bin/env python3
"""
CLI for querying the configured vector store (vector mode by default).

Usage:
    python scripts/search.py --q "reset my password" --collection dental --mode vector --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Ensure project root is on sys.path so `from src...` imports work when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_TOP_K,
    ENABLE_RERANK,
    RERANK_BACKEND,
    RERANK_MODEL,
    VECTOR_STORE_PROVIDER,
)  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.retriever import retrieve  # noqa: E402
from src.reranker import RerankerConfig, rerank_hits  # noqa: E402
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
    parser.add_argument(
        "--trace",
        default=None,
        help="If provided, append a JSONL trace of the retrieval results to this path.",
    )
    parser.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        default=ENABLE_RERANK,
        help=f"Enable reranking (default: {ENABLE_RERANK}).",
    )
    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable reranking (overrides --rerank).",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=DEFAULT_RERANK_DEPTH,
        help=f"Depth of hits to rerank (default: {DEFAULT_RERANK_DEPTH}).",
    )
    parser.add_argument(
        "--rerank-backend",
        default=RERANK_BACKEND,
        help=f"Rerank backend identifier (default: {RERANK_BACKEND}).",
    )
    parser.add_argument(
        "--rerank-model",
        default=RERANK_MODEL,
        help=f"Rerank model identifier (default: {RERANK_MODEL}).",
    )
    return parser.parse_args(argv)


def _write_trace(
    trace_path: str,
    query: str,
    args: argparse.Namespace,
    hits: List[dict],
) -> None:
    """Persist a JSONL record of the retrieval for labeling/debugging."""
    path = Path(trace_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    top_hits: List[dict] = []
    for hit in hits[: args.k]:
        top_hits.append(
            {
                "id": hit.get("id"),
                "score": float(hit.get("score", 0.0) or 0.0),
                "title": hit.get("title"),
                "source": hit.get("source"),
                "text": hit.get("text"),
            }
        )

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "params": {
            "mode": args.mode,
            "k": args.k,
            "alpha": args.alpha,
            "model": args.model,
            "collection": args.collection,
            "rerank": bool(getattr(args, "rerank", False)),
            "rerank_depth": getattr(args, "rerank_depth", None),
            "rerank_backend": getattr(args, "rerank_backend", None),
            "rerank_model": getattr(args, "rerank_model", None),
        },
        "hits": top_hits,
    }
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record, fh)
        fh.write("\n")


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

    reranked_hits = hits
    if args.rerank:
        rerank_config = RerankerConfig(
            enabled=True,
            backend=args.rerank_backend,
            model_name=args.rerank_model,
            depth=args.rerank_depth,
        )
        reranked_hits = rerank_hits(query=args.q, hits=hits, config=rerank_config)

    if args.trace:
        _write_trace(args.trace, args.q, args, reranked_hits)

    if not reranked_hits:
        print("No results.")
        return 0

    print(f"Top {min(len(reranked_hits), args.k)} results for query '{args.q}' (mode={args.mode}):")
    for idx, hit in enumerate(reranked_hits[: args.k], start=1):
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
