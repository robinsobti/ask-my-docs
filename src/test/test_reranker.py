from __future__ import annotations

from typing import List, Dict

import pytest

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reranker import RerankerConfig, rerank_hits

def _make_hits() -> List[Dict[str, object]]:
    return [
        {"id": "doc::chunk#0", "score": 0.8, "title": "doc", "text": "first hit", "source": "a"},
        {"id": "doc::chunk#1", "score": 0.6, "title": "doc", "text": "second hit", "source": "a"},
        {"id": "doc::chunk#2", "score": 0.4, "title": "doc", "text": "third hit", "source": "a"},
    ]

def test_rerank_disabled_returns_original() -> None:
    hits = _make_hits()
    cfg = RerankerConfig(enabled=False)
    assert rerank_hits("query", hits, cfg) == hits


def test_rerank_orders_by_backend_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_backend(query: str, hits, config) -> List[float]:
        calls["invoked"] = True
        assert query == "question"
        assert len(hits) == 3
        return [0.2, 0.9, 0.5]

    monkeypatch.setattr("src.reranker._score_with_backend", fake_backend, raising=True)
    hits = _make_hits()
    cfg = RerankerConfig(enabled=True, backend="hf", depth=3)
    reranked = rerank_hits("question", hits, cfg)
    assert calls["invoked"]
    assert [hit["id"] for hit in reranked] == ["doc::chunk#1", "doc::chunk#2", "doc::chunk#0"]
    for hit in reranked[:3]:
        assert "rerank_score" in hit
        assert hit["rerank_backend"] == "hf"


def test_rerank_timeout_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_backend(*args, **kwargs):
        raise TimeoutError("Simulated timeout")

    monkeypatch.setattr("src.reranker._score_with_backend", slow_backend, raising=True)
    hits = _make_hits()
    cfg = RerankerConfig(enabled=True, backend="hf", depth=3)
    reranked = rerank_hits("question", hits, cfg)
    assert reranked == hits
