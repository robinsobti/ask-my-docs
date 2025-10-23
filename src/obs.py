from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, MutableMapping


def init_run_log() -> tuple[str, Path]:
    """
    Initialize a run directory and the queries.jsonl log file.
    """
    base_dir = Path("runs")
    base_dir.mkdir(parents=True, exist_ok=True)

    run_identifier = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M-%S")
    run_dir = base_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    queries_path = run_dir / "queries.jsonl"
    queries_path.touch(exist_ok=True)

    latest_path = base_dir / "latest"
    target = run_dir.resolve()
    try:
        if latest_path.exists() or latest_path.is_symlink():
            if latest_path.is_symlink() or latest_path.is_file():
                latest_path.unlink()
            else:
                shutil.rmtree(latest_path)
        latest_path.symlink_to(target, target_is_directory=True)
    except OSError:
        # Fallback to a mirrored directory when symlinks are not supported.
        if latest_path.exists():
            if latest_path.is_symlink() or latest_path.is_file():
                latest_path.unlink()
            else:
                shutil.rmtree(latest_path)
        shutil.copytree(target, latest_path)
    return run_identifier, queries_path.resolve()


def new_ctx(run_id: str, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a mutable context dict used to accumulate observability metadata.
    """
    return {
        "run_id": run_id,
        "query": query,
        "params": dict(params),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "top_docs": [],
        "reranked_docs": [],
        "answer": None,
        "answer_chars": None,
        "answer_citations": [],
        "token_usage": None,
        "cost_usd": None,
        "latency_ms": None,
        "error": None,
    }

@contextmanager
def timer(stage: str, ctx: MutableMapping[str, Any]) -> Iterator[None]:
    """
    Context manager that records elapsed time for a named stage.
    """
    if ctx is None:
        raise ValueError("ctx is required for timing.")

    stages = ctx.setdefault("stages", {})
    start_dt = datetime.now(timezone.utc)
    start_perf = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (perf_counter() - start_perf) * 1000.0
        end_dt = datetime.now(timezone.utc)
        stages[stage] = {
            "started_at": start_dt.isoformat(),
            "ended_at": end_dt.isoformat(),
            "elapsed_ms": round(elapsed_ms, 3),
        }
        ctx.setdefault("stages", stages)


def append_query_log(path: Path, record: Dict[str, Any]) -> None:
    """
    Append a single JSONL record to the provided path.
    """
    normalized_path = path.expanduser()
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=True)
    with normalized_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def record_error(ctx: MutableMapping[str, Any], exc: Exception | str) -> None:
    """
    Store error information on the context for downstream logging.
    """
    if isinstance(exc, Exception):
        message = str(exc)
        error_type = type(exc).__name__
    else:
        message = str(exc)
        error_type = "Error"
    ctx["error"] = {
        "type": error_type,
        "message": message,
    }
