from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Type aliases used throughout the module.
GoldRecord = Dict[str, Any]
Hit = Dict[str, Any]
RunFn = Callable[[str, int], List[Hit]]

@dataclass
class MetricResult:
    """
    Structure for storing metric values per run.

    TODO: populate this dataclass with the fields you need once evaluate_retrieval
    returns concrete metrics (e.g., hit@k values). For now it simply mirrors a
    dict but keeps the type explicit for future refactoring.
    """

    name: str
    metrics: Dict[str, float]


def load_goldset(path: str) -> List[GoldRecord]:
    """
    Read goldset entries from a JSONL file.

    TODO:
    - Open the JSONL, skip blank lines, parse each line into a dict.
    - Validate required keys (`question`, `answers`); coerce answers to a list.
    - Consider storing an `id` or index for traceability.
    - Return a list of records preserving file order.
    """
    goldset_path = Path(path)
    if not goldset_path.exists():
        raise FileNotFoundError(f"Goldset path '{goldset_path}' does not exist.")

    records: List[GoldRecord] = []
    with goldset_path.open("r", encoding="utf-8") as fh:
        for index, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON on line {index}: {line}") from exc

            if not isinstance(entry, dict):
                raise ValueError(f"Goldset entry on line {index} is not an object: {entry!r}")

            question = entry.get("question")
            if not question or not isinstance(question, str):
                raise ValueError(f"Goldset entry on line {index} missing 'question': {entry!r}")

            answers = entry.get("answers", [])
            if answers is None:
                answers = []
            if not isinstance(answers, list):
                answers = [answers]

            normalized_answers: List[str] = []
            for ans in answers:
                if ans is None:
                    continue
                if not isinstance(ans, str):
                    raise ValueError(f"Goldset answer on line {index} must be string, got {ans!r}")
                normalized_answers.append(ans)

            notes = entry.get("notes", "")
            if notes is None:
                notes = ""
            if not isinstance(notes, str):
                notes = str(notes)

            record = {
                "id": entry.get("id", f"gold-{index}"),
                "question": question.strip(),
                "answers": normalized_answers,
                "notes": notes.strip(),
            }
            records.append(record)

    if not records:
        raise ValueError(f"Goldset at '{goldset_path}' produced no records.")

    return records
        
def evaluate_retrieval(
    queries: List[GoldRecord],
    run_fn: RunFn,
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Evaluate a single retriever configuration on the provided queries.

    Iterates through each goldset record, runs the supplied callable to obtain
    retrieval hits, and computes Hit@K, Recall@K, and MRR. Metrics are averaged
    across all records and returned as a dictionary.
    """
    if not queries:
        raise ValueError("evaluate_retrieval requires at least one query.")
    if not ks:
        raise ValueError("Parameter 'ks' must contain at least one value.")

    normalized_ks = sorted({int(k) for k in ks if int(k) > 0})
    if not normalized_ks:
        raise ValueError("All values in 'ks' must be positive integers.")
    max_k = max(normalized_ks)

    totals: Dict[str, float] = {f"hit@{k}": 0.0 for k in normalized_ks}
    totals.update({f"recall@{k}": 0.0 for k in normalized_ks})
    totals["mrr"] = 0.0

    for record in queries:
        question = record.get("question", "").strip()
        answers = record.get("answers", [])
        answer_set = {ans for ans in answers if isinstance(ans, str)}

        hits = run_fn(question, max_k) or []
        hit_ids: List[str] = []
        for hit in hits:
            hit_id = hit.get("id")
            if isinstance(hit_id, str):
                hit_ids.append(hit_id)

        is_no_answer = not answer_set

        if is_no_answer:
            has_hits = bool(hit_ids)
            for k in normalized_ks:
                success = 0.0 if has_hits else 1.0
                totals[f"hit@{k}"] += success
                totals[f"recall@{k}"] += success
            totals["mrr"] += 0.0 if has_hits else 1.0
            continue

        for k in normalized_ks:
            top_ids = hit_ids[:k]
            relevant_in_topk = answer_set.intersection(top_ids)
            totals[f"hit@{k}"] += 1.0 if relevant_in_topk else 0.0
            totals[f"recall@{k}"] += len(relevant_in_topk) / len(answer_set)

        reciprocal_rank = 0.0
        for rank, hit_id in enumerate(hit_ids, start=1):
            if hit_id in answer_set:
                reciprocal_rank = 1.0 / rank
                break
        totals["mrr"] += reciprocal_rank

    count = float(len(queries))
    return {metric: value / count for metric, value in totals.items()}

def compare_runs(
    queries: List[GoldRecord],
    run_fns: Dict[str, RunFn],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> Dict[str, Dict[str, float]]:
    """
    Run multiple retriever variants and collate their metrics.

    Executes evaluate_retrieval for every entry in run_fns and returns a
    dictionary of results keyed by run name.
    """
    if not run_fns:
        raise ValueError("No run functions supplied for comparison.")

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in run_fns.items():
        if not callable(fn):
            raise ValueError(f"Run function for '{name}' is not callable.")
        results[name] = evaluate_retrieval(queries, fn, ks=ks)
    return results


def format_delta_table(
    results: Dict[str, Dict[str, float]],
    *,
    baseline: str,
) -> str:
    """
    Produce a human-readable table comparing each variant to the baseline.

    Creates a plain-text table with columns (metric, baseline, run, delta) for
    each non-baseline run.
    """
    if baseline not in results:
        raise KeyError(f"Baseline run '{baseline}' not found in results.")

    baseline_metrics = results[baseline]
    lines: List[str] = []
    header = f"{'metric':<20} {'baseline':>10} {'run':>10} {'delta':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for run_name, metrics in results.items():
        if run_name == baseline:
            continue
        for metric_name, value in metrics.items():
            base_value = baseline_metrics.get(metric_name, 0.0)
            delta = value - base_value
            lines.append(
                f"{metric_name:<20} {base_value:>10.4f} {value:>10.4f} {delta:>10.4f}"
            )

    if len(lines) == 2:
        lines.append("(no variant runs to compare)")

    return "\n".join(lines)


def sample_queries_from_logs(
    log_paths: Sequence[str],
    *,
    limit: Optional[int] = None,
) -> List[GoldRecord]:
    """
    Load queries from existing run logs for drift detection.

    Parses JSONL query logs, extracts unique query strings, and returns a list
    of goldset-style records (empty answers) limited to `limit` entries if
    specified.
    """
    seen: set[str] = set()
    records: List[GoldRecord] = []
    remaining = limit if limit is None else max(limit, 0)

    for raw_path in log_paths:
        if remaining is not None and remaining <= 0:
            break
        path = Path(raw_path)
        if not path.exists():
            continue

        for entry in _iter_jsonl(path):
            query = entry.get("query")
            if not query or not isinstance(query, str):
                continue
            normalized_query = query.strip()
            if not normalized_query or normalized_query in seen:
                continue
            seen.add(normalized_query)
            record = {
                "id": entry.get("id", normalized_query),
                "question": normalized_query,
                "answers": [],
                "notes": entry.get("notes", ""),
            }
            records.append(record)
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break

    return records


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Utility generator to yield JSON objects from a JSONL file.

    Yields dictionaries read from the provided file path. Lines that fail to
    parse raise ValueError to surface malformed logs early.
    """
    with path.open("r", encoding="utf-8") as fh:
        for index, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {index}: {line}") from exc
            if isinstance(entry, dict):
                yield entry
            else:
                raise ValueError(f"Expected JSON object in {path} on line {index}, got {entry!r}")


__all__ = [
    "GoldRecord",
    "Hit",
    "MetricResult",
    "load_goldset",
    "evaluate_retrieval",
    "compare_runs",
    "format_delta_table",
    "sample_queries_from_logs",
]
