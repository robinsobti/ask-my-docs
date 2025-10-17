import numpy as np
import pytest

import src.retriever as retriever
from src.retriever import fuse_hybrid, retrieve


class DummyEmbedder:
    def __init__(self, dim: int = 3) -> None:
        self._dim = dim

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        vectors = np.ones((len(texts), self._dim), dtype=np.float32)
        if normalize and len(texts) > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        return vectors

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture(autouse=True)
def stub_weaviate_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid touching a real Weaviate instance during unit tests."""
    monkeypatch.setattr(retriever, "create_collection_if_missing", lambda *args, **kwargs: None)
    monkeypatch.setattr(retriever, "search_bm25", lambda *args, **kwargs: [])
    monkeypatch.setattr(retriever, "search_vector", lambda *args, **kwargs: [])


def test_empty_query_returns_empty_list() -> None:
    embedder = DummyEmbedder()
    assert retrieve("", mode="bm25", k=5, alpha=0.5, embedder=embedder) == []


def test_invalid_args_raise() -> None:
    embedder = DummyEmbedder()
    with pytest.raises(ValueError):
        retrieve("hi", mode="nope", k=5, alpha=0.5, embedder=embedder)
    with pytest.raises(ValueError):
        retrieve("hi", mode="bm25", k=0, alpha=0.5, embedder=embedder)


def test_vector_mode_requires_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retriever, "create_collection_if_missing", lambda *args, **kwargs: None)
    with pytest.raises(ValueError):
        retrieve("hi", mode="vector", k=5, alpha=0.5, embedder=None)


def test_fuse_hybrid_score_monotonic() -> None:
    bm25_hits = [
        {"id": "a", "score": 0.8, "title": "A", "text": "A text", "source": "srcA"},
        {"id": "b", "score": 0.4, "title": "B", "text": "B text", "source": "srcB"},
    ]
    vector_hits = [
        {"id": "a", "score": 0.3, "title": "A", "text": "A text", "source": "srcA"},
        {"id": "c", "score": 0.9, "title": "C", "text": "C text", "source": "srcC"},
    ]

    all_bm25 = fuse_hybrid(bm25_hits, vector_hits, alpha=1.0)
    assert all_bm25[0]["id"] == "a"
    assert pytest.approx(all_bm25[0]["score"], rel=1e-6) == 0.8

    all_vector = fuse_hybrid(bm25_hits, vector_hits, alpha=0.0)
    assert all_vector[0]["id"] == "c"
    assert pytest.approx(all_vector[0]["score"], rel=1e-6) == 0.9

    with pytest.raises(ValueError):
        fuse_hybrid([], [], alpha=1.5)
