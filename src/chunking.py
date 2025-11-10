from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List
import os
import re
import uuid

from PyPDF2 import PdfReader

from .config import (
    CHUNK_BOUNDARY_WINDOW,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict | None = None


@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    chunk_index: int
    chunk_id: str
    metadata: dict | None = None


def _slugify(value: str, *, fallback: str = "doc") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    slug = slug.strip("-") or fallback
    return slug[:48]


def _read_file(fpath: Path) -> str | None:
    ext = fpath.suffix.lower()
    try:
        if ext in {".txt", ".md"}:
            text = fpath.read_text(encoding="utf-8")
            return text.strip() if text and len(text.strip()) > 50 else None
        if ext == ".pdf":
            reader = PdfReader(str(fpath))
            pages: List[str] = []
            for idx, page in enumerate(reader.pages):
                chunk = (page.extract_text() or "").strip()
                if idx < 2:  # drop cover + front-matter
                    continue
                if len(chunk) < 50:
                    continue
                pages.append(chunk)
            return "\n".join(pages) if pages else None
    except Exception:
        return None
    return None


def _meta_for(fpath: Path) -> dict:
    base_name = fpath.stem
    display = base_name.replace("_", " ").strip()
    return {
        "source": fpath.name,
        "filename": fpath.name,
        "book_name": base_name,
        "display_title": display or base_name,
    }


def load_docs(path: str) -> List[Document]:
    """Read docs from a directory or a single file path."""
    documents: List[Document] = []
    root_path = Path(path).resolve()
    existing_ids: set[str] = set()

    def _assign_id(base: str) -> str:
        base_slug = _slugify(base)
        doc_id = base_slug
        suffix = 1
        while doc_id in existing_ids:
            suffix += 1
            doc_id = f"{base_slug}-{suffix}"
        existing_ids.add(doc_id)
        return doc_id

    if root_path.is_file():
        text = _read_file(root_path)
        if text:
            documents.append(
                Document(id=_assign_id(root_path.stem), content=text, metadata=_meta_for(root_path))
            )
        return documents

    for root, _, files in os.walk(root_path):
        files.sort()
        for file in files:
            fpath = Path(root) / file
            text = _read_file(fpath)
            if not text:
                continue
            documents.append(Document(id=_assign_id(fpath.stem), content=text, metadata=_meta_for(fpath)))
    return documents


def load_files(paths: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield raw docs with basic metadata."""
    for path in paths:
        for doc in load_docs(path):
            meta = doc.metadata or {}
            slim_meta = {k: meta.get(k) for k in ("book_name", "filename") if meta.get(k)}
            yield {
                "doc_id": doc.id,
                "title": meta.get("display_title") or meta.get("book_name") or meta.get("filename") or doc.id,
                "source": meta.get("filename") or "",
                "text": doc.content,
                "metadata": slim_meta,
            }


def split_into_chunks(
    doc: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    boundary_window: int = CHUNK_BOUNDARY_WINDOW,
) -> Iterator[Dict[str, Any]]:
    """
    Yield chunks with whitespace-aware boundaries.
    """
    doc_id = str(doc.get("doc_id", "")).strip() or "doc"
    title = str(doc.get("title", "") or "")
    source = str(doc.get("source", "") or "")
    text = str(doc.get("text", "") or "").strip()
    metadata = dict(doc.get("metadata") or {})

    if not text:
        return
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    n = len(text)
    start = 0
    idx = 0

    def _snap_boundary(s: int, e: int, max_look: int = boundary_window) -> int:
        fwd = e
        limit = min(n, e + max_look)
        while fwd < limit and not text[fwd].isspace():
            fwd += 1
        if fwd < n and fwd != e:
            return fwd
        back = e
        limit = max(s, e - max_look)
        while back > limit and not text[back - 1].isspace():
            back -= 1
        if back > s:
            return back
        return e

    while start < n:
        end = min(n, start + chunk_size)
        end = _snap_boundary(start, end)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{doc_id}#{idx}"
            yield {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "content": chunk_text,
                "text": chunk_text,
                "title": title,
                "source": source,
                "ord": idx,
                "metadata": metadata,
            }
            idx += 1
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start


def chunk_docs(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    splitter: str = "recursive",
    boundary_window: int = CHUNK_BOUNDARY_WINDOW,
) -> List[Chunk]:
    """Produce deterministic chunk_ids; return list of Chunk dataclasses."""
    chunks: List[Chunk] = []
    for doc in docs:
        doc_dict = {
            "doc_id": doc.id,
            "title": doc.metadata.get("display_title") if doc.metadata else "",
            "source": doc.metadata.get("filename") if doc.metadata else "",
            "text": doc.content,
            "metadata": dict(doc.metadata) if doc.metadata else {},
        }
        parts = list(
            split_into_chunks(
                doc_dict,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                boundary_window=boundary_window,
            )
        )
        for c in parts:
            idx = c["ord"]
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}::{idx}"))
            chunk_meta = dict(doc.metadata) if doc.metadata else {}
            chunk_meta.update({"title": c["title"], "source": c["source"]})
            chunks.append(
                Chunk(
                    id=cid,
                    content=c["content"],
                    doc_id=doc.id,
                    chunk_index=idx,
                    chunk_id=c["chunk_id"],
                    metadata=chunk_meta,
                )
            )
    return chunks
