from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import (
    DEFAULT_RERANK_DEPTH,
    ENABLE_RERANK,
    RERANK_BACKEND,
    RERANK_MODEL,
    RERANK_BATCH_SIZE,
    RERANK_TIMEOUT_S,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

Hit = Dict[str, object]

_CROSS_ENCODER_CACHE: Dict[str, object] = {}


@dataclass
class RerankerConfig:
    """Runtime settings controlling rerank behaviour."""

    enabled: bool = ENABLE_RERANK
    backend: str = RERANK_BACKEND
    model_name: str = RERANK_MODEL
    depth: int = DEFAULT_RERANK_DEPTH
    batch_size: int = RERANK_BATCH_SIZE
    timeout_s: float = RERANK_TIMEOUT_S
    api_key: Optional[str] = OPENAI_API_KEY or None

    def clamp_depth(self, available: int) -> int:
        if self.depth <= 0:
            return available
        return min(self.depth, available)

    def effective_enabled(self) -> bool:
        return self.enabled and self.backend not in {"none", "", None}


def rerank_hits(
    query: str,
    hits: Sequence[Hit],
    config: Optional[RerankerConfig] = None,
) -> List[Hit]:
    """
    Return hits sorted by reranker score while preserving stable ids.

    Falls back to the original ordering if reranking is disabled or the backend
    raises an error/timeout. Tail hits beyond `depth` are appended unchanged.
    """
    if not hits:
        return []

    cfg = config or RerankerConfig()
    if not cfg.effective_enabled():
        return list(hits)

    depth = cfg.clamp_depth(len(hits))
    if depth <= 1:
        return list(hits)

    head = list(hits[:depth])
    tail = list(hits[depth:])

    try:
        scores = _score_with_backend(query=query, hits=head, config=cfg)
    except NotImplementedError as exc:
        logger.warning("Rerank backend %s not implemented: %s", cfg.backend, exc)
        return list(hits)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Rerank backend %s failed: %s", cfg.backend, exc)
        return list(hits)

    if not scores:
        logger.warning("Rerank backend %s returned no scores; keeping original order.", cfg.backend)
        return list(hits)
    if len(scores) != len(head):
        logger.warning(
            "Rerank backend %s returned %s scores for %s hits; keeping original order.",
            cfg.backend,
            len(scores),
            len(head),
        )
        return list(hits)

    ranked_pairs: List[Tuple[float, int, Hit]] = []
    for idx, (hit, score) in enumerate(zip(head, scores)):
        new_score = float(score)
        updated = dict(hit)
        updated["score"] = new_score
        updated["rerank_score"] = new_score
        updated["rerank_backend"] = cfg.backend
        ranked_pairs.append((new_score, idx, updated))

    ranked_pairs.sort(key=lambda entry: (entry[0], -entry[1]), reverse=True)
    reranked_head = [entry[2] for entry in ranked_pairs]

    reranked = reranked_head + list(tail)
    return reranked


def _score_with_backend(query: str, hits: Sequence[Hit], config: RerankerConfig) -> List[float]:
    """Dispatch scoring to the configured backend."""
    backend = (config.backend or "none").lower()
    if backend in {"hf", "huggingface", "cross-encoder", "hf-cross-encoder"}:
        return _score_cross_encoder(query=query, hits=hits, config=config)
    if backend in {"openai", "openai-api"}:
        return _score_openai_api(query=query, hits=hits, config=config)
    if backend in {"cohere", "cohere-api"}:
        return _score_cohere_api(query=query, hits=hits, config=config)
    raise NotImplementedError(f"Backend '{backend}' is not supported.")


def _score_cross_encoder(query: str, hits: Sequence[Hit], config: RerankerConfig) -> List[float]:
    """Score hits using a HuggingFace cross-encoder."""
    if not hits:
        return []

    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("sentence-transformers is required for cross-encoder reranking.") from exc

    model = _CROSS_ENCODER_CACHE.get(config.model_name)
    if model is None:
        model = CrossEncoder(config.model_name)
        _CROSS_ENCODER_CACHE[config.model_name] = model

    timeout_limit = float(config.timeout_s or 0.0)
    start = time.perf_counter()
    scores: List[float] = []

    for batch in _batched_pairs(query, hits, max(1, config.batch_size)):
        if timeout_limit > 0 and (time.perf_counter() - start) > timeout_limit:
            raise TimeoutError("Cross-encoder rerank exceeded timeout.")
        batch_scores = model.predict(batch)
        scores.extend(float(val) for val in batch_scores)
        if timeout_limit > 0 and (time.perf_counter() - start) > timeout_limit:
            raise TimeoutError("Cross-encoder rerank exceeded timeout.")

    return scores[: len(hits)]


def _score_openai_api(query: str, hits: Sequence[Hit], config: RerankerConfig) -> List[float]:
    """Score hits using OpenAI embeddings similarity as a proxy reranker."""
    if not hits:
        return []
    if not config.api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI reranking.")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("openai package is required for OpenAI reranking.") from exc

    client = OpenAI(api_key=config.api_key, timeout=config.timeout_s or None)
    documents = [str(hit.get("text") or "") for hit in hits]
    inputs = [query] + documents
    response = client.embeddings.create(model=config.model_name, input=inputs)
    embeddings = [item.embedding for item in response.data]
    if len(embeddings) != len(inputs):
        raise RuntimeError("OpenAI embeddings response length mismatch.")

    query_vec = embeddings[0]
    doc_vecs = embeddings[1:]
    scores = [_cosine_similarity(query_vec, vec) for vec in doc_vecs]
    return scores


def _score_cohere_api(query: str, hits: Sequence[Hit], config: RerankerConfig) -> List[float]:
    """Score hits using Cohere's rerank API."""
    if not hits:
        return []
    if not config.api_key:
        raise RuntimeError("COHERE_API_KEY is required for Cohere reranking.")

    try:
        import cohere  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("cohere package is required for Cohere reranking.") from exc

    client = cohere.Client(api_key=config.api_key, timeout=config.timeout_s or None)
    documents = [str(hit.get("text") or "") for hit in hits]
    response = client.rerank(
        query=query,
        documents=documents,
        model=config.model_name,
        top_n=len(documents),
    )
    score_map = {result.index: float(result.relevance_score) for result in response.results}
    scores = [score_map.get(idx, 0.0) for idx in range(len(documents))]
    return scores


def _batched_pairs(
    query: str,
    hits: Sequence[Hit],
    batch_size: int,
) -> Iterable[List[Tuple[str, str]]]:
    """
    Yield batches of (query, text) pairs for model inference.

    This helper keeps batching logic central so all backends can share it.
    """
    batch: List[Tuple[str, str]] = []
    for hit in hits:
        text = str(hit.get("text") or "")
        batch.append((query, text))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(float(b) ** 2 for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "RerankerConfig",
    "rerank_hits",
]
