#!/usr/bin/env python3
"""
CLI for running retrieval/rerank evaluations against the goldset.

Implementation scaffolding only; follow the instructions inside `main()` to wire up
argument parsing, evaluation, and reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is importable when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RETRIEVAL_TOP_K,
    ENABLE_RERANK,
    VECTOR_STORE_PROVIDER,
)
from src.evals import (
    compare_runs,
    format_delta_table,
    load_goldset,
    sample_queries_from_logs,
)
from src.embedder import Embedder
from src.reranker import RerankerConfig, rerank_hits
from src.retriever import retrieve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval configurations.")
    parser.add_argument(
        "--goldset",
        default=str(PROJECT_ROOT / "data" / "goldset.jsonl"),
        help="Path to goldset JSONL file.",
    )
    parser.add_argument(
        "--from-logs",
        nargs="*",
        help="Optional list of run log JSONL files to sample queries from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries when using --from-logs.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help="Vector store collection/index to query.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "runs"),
        help="Directory where eval results should be written.",
    )
    default_modes = ["vector", "hybrid"] if VECTOR_STORE_PROVIDER == "weaviate" else ["vector"]
    parser.add_argument(
        "--modes",
        nargs="*",
        default=default_modes,
        help="Retrieval modes to evaluate (baseline set).",
    )
    parser.add_argument(
        "--enable-rerank",
        dest="enable_rerank",
        action="store_true",
        default=ENABLE_RERANK,
        help=f"Enable reranking (default: {ENABLE_RERANK}).",
    )
    parser.add_argument(
        "--disable-rerank",
        dest="enable_rerank",
        action="store_false",
        help="Disable reranking for all runs.",
    )
    parser.add_argument(
        "--rerank-backend",
        default=None,
        help="Override rerank backend (defaults to config).",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Override rerank model name (defaults to config).",
    )
    parser.add_argument(
        "--rerank-depth",
        type=int,
        default=None,
        help="Override rerank depth (defaults to config).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_RETRIEVAL_TOP_K,
        help="Number of documents to request from the retriever.",
    )
    return parser


def _run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Execute evaluation for the provided arguments and return summary details.
    """
    # Load queries from logs (if provided) or the goldset.
    queries: List[Dict[str, Any]] = []
    if args.from_logs:
        queries = sample_queries_from_logs(args.from_logs, limit=args.limit)
    if not queries:
        queries = load_goldset(args.goldset)
    if not queries:
        raise ValueError("No queries available for evaluation.")

    # Normalise modes and ensure uniqueness while preserving order.
    modes: List[str] = []
    seen_modes: set[str] = set()
    for mode in args.modes or []:
        normalized = mode.lower()
        if normalized and normalized not in seen_modes:
            modes.append(normalized)
            seen_modes.add(normalized)
    if not modes:
        raise ValueError("At least one retrieval mode must be provided.")

    filtered_modes: List[str] = []
    skipped_modes: List[str] = []
    for mode in modes:
        if mode == "hybrid" and VECTOR_STORE_PROVIDER != "weaviate":
            skipped_modes.append(mode)
            continue
        if mode == "bm25" and VECTOR_STORE_PROVIDER != "weaviate":
            skipped_modes.append(mode)
            continue
        filtered_modes.append(mode)
    if skipped_modes:
        warnings.warn(
            f"Skipping unsupported modes for provider '{VECTOR_STORE_PROVIDER}': {', '.join(skipped_modes)}",
            RuntimeWarning,
        )
    modes = filtered_modes
    if not modes:
        raise ValueError("No supported retrieval modes remain after filtering.")

    # Prepare reranker configuration if requested.
    rerank_config: Optional[RerankerConfig] = None
    if args.enable_rerank:
        base_cfg = RerankerConfig()
        rerank_config = RerankerConfig(
            enabled=True,
            backend=args.rerank_backend or base_cfg.backend,
            model_name=args.rerank_model or base_cfg.model_name,
            depth=args.rerank_depth or base_cfg.depth,
            batch_size=base_cfg.batch_size,
            timeout_s=base_cfg.timeout_s,
            api_key=base_cfg.api_key,
        )
        if rerank_config.backend in {"none", "", None}:
            raise ValueError(
                "Rerank enabled but no backend configured. Provide --rerank-backend or set RERANK_BACKEND."
            )

    # Cache embedder instances per model to avoid re-instantiation.
    embedder_cache: Dict[str, Embedder] = {}

    def _get_embedder() -> Embedder:
        model_name = DEFAULT_EMBEDDER_MODEL
        if model_name not in embedder_cache:
            embedder_cache[model_name] = Embedder(model_name=model_name)
        return embedder_cache[model_name]

    def _make_run_fn(mode: str, use_rerank: bool) -> Callable[[str, int], List[Dict[str, Any]]]:
        def _run(query: str, top_k: int) -> List[Dict[str, Any]]:
            request_k = max(top_k, args.top_k)
            embedder = None
            if mode in {"vector", "hybrid"}:
                embedder = _get_embedder()
            hits = retrieve(
                query=query,
                mode=mode,
                k=request_k,
                alpha=DEFAULT_HYBRID_ALPHA,
                collection=args.collection,
                embedder=embedder,
            )
            if use_rerank and rerank_config is not None:
                return rerank_hits(query=query, hits=hits, config=rerank_config)
            return hits

        return _run

    run_functions: Dict[str, Callable[[str, int], List[Dict[str, Any]]]] = {}
    for mode in modes:
        run_functions[mode] = _make_run_fn(mode, use_rerank=False)
        if rerank_config is not None:
            run_functions[f"{mode}+rerank"] = _make_run_fn(mode, use_rerank=True)

    if not run_functions:
        raise ValueError("No run functions were created for evaluation.")

    results = compare_runs(queries, run_functions)
    baseline = modes[0]

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M-%S")
    output_root = Path(args.output_dir)
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": timestamp,
        "collection": args.collection,
        "query_count": len(queries),
        "modes": modes,
        "runs_evaluated": list(run_functions.keys()),
        "rerank_enabled": bool(rerank_config),
        "metrics": results,
    }
    eval_path = run_dir / "eval.json"
    with eval_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    table = format_delta_table(results, baseline=baseline)
    table_path = run_dir / "eval_table.txt"
    table_path.write_text(table + "\n", encoding="utf-8")

    return {
        "payload": payload,
        "table": table,
        "eval_path": eval_path,
        "table_path": table_path,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the evaluation CLI.

    Steps:
    1. Parse arguments.
    2. Load queries (goldset or sampled logs).
    3. Build run functions for each retrieval mode (+ optional rerank).
    4. Run evaluations and persist results.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        result = _run_evaluation(args)
        table = result["table"]
        eval_path = result["eval_path"]
        table_path = result["table_path"]

        print(table)
        print(f"\nSaved metrics to {eval_path}")
        print(f"Saved delta table to {table_path}")
        return 0

    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


def run_eval_with_defaults() -> Dict[str, Any]:
    """
    Convenience helper to run the evaluation using default arguments.

    Returns the same payload dictionary produced by the CLI helper.
    """
    parser = _build_parser()
    args = parser.parse_args([])
    return _run_evaluation(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        # Close any retriever/vector-store connections if needed in the future.
        pass
