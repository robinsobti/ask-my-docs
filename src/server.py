"""FastAPI service exposing retrieval capabilities and a health endpoint."""
from __future__ import annotations

import os
import threading
from typing import List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    COLLECTION_NAME,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_TOP_K,
)
from src.embedder import Embedder
from src.retriever import retrieve

BUILD_INFO = {
    "version": os.getenv("APP_BUILD_VERSION", "dev"),
    "sha": os.getenv("APP_BUILD_SHA", "local"),
    "time": os.getenv("APP_BUILD_TIME", "unknown"),
}

app = FastAPI(title="Ask-My-Docs API", version=BUILD_INFO["version"])

_embedder_lock = threading.Lock()
_embedder: Embedder | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to feed the retriever.")
    mode: Literal["bm25", "vector", "hybrid"] = Field(
        DEFAULT_RETRIEVAL_MODE,
        description="Retrieval backend to use.",
    )
    k: int = Field(
        DEFAULT_RETRIEVAL_TOP_K,
        ge=1,
        le=50,
        description="Number of hits to return (enforced max = 50).",
    )
    alpha: float = Field(
        DEFAULT_HYBRID_ALPHA,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting factor (ignored outside hybrid mode).",
    )
    collection: str = Field(
        COLLECTION_NAME,
        description="Vector store collection/namespace to query.",
    )


class RetrievalResult(BaseModel):
    id: str
    title: str = ""
    text: str = ""
    source: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    count: int
    results: List[RetrievalResult]


def _embedder_for_mode(mode: str) -> Embedder | None:
    """Instantiate the embedder only when vector-backed retrieval is requested."""

    normalized = mode.lower()
    if normalized == "bm25":
        return None

    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder


def _normalize_hit(hit: dict) -> RetrievalResult:
    return RetrievalResult(
        id=str(hit.get("id", "")),
        title=str(hit.get("title") or ""),
        text=str(hit.get("text") or ""),
        source=str(hit.get("source") or ""),
        score=float(hit.get("score") or 0.0),
    )


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "ask-my-docs", "status": "ok"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "build": BUILD_INFO}


@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest) -> QueryResponse:
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        hits = retrieve(
            query_text,
            mode=request.mode,
            k=request.k,
            alpha=request.alpha,
            collection=request.collection,
            embedder=_embedder_for_mode(request.mode),
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [_normalize_hit(hit) for hit in hits]
    return QueryResponse(count=len(results), results=results)
