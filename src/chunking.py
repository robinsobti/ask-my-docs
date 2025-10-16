from typing import List, Iterator, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import os
import re
import uuid
from PyPDF2 import PdfReader

@dataclass
class Document:
    id: str
    content: str
    metadata: dict = None

@dataclass
class Chunk:
    id: str
    content: str
    doc_id: str
    chunk_index: int
    chunk_id: str
    metadata: dict = None

def load_docs(path: str) -> List[Document]:
    """Walk a folder; read .txt/.md directly; extract text from .pdf (basic)."""
    documents: List[Document] = []
    for root, _, files in os.walk(path):
        for file in files:
            fpath = os.path.join(root, file)
            ext = Path(file).suffix.lower()
            try:
                if ext in (".txt", ".md"):
                    with open(fpath, "r", encoding="utf-8") as fh:
                        text = fh.read()
                elif ext == ".pdf":
                    reader = PdfReader(fpath)
                    pages = []
                    for p in reader.pages:
                        t = p.extract_text()
                        if t:
                            pages.append(t)
                    text = "\n".join(pages)
                else:
                    # skip unknown file types
                    continue
            except Exception:
                # skip files we can't read
                continue

            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, fpath))
            meta = {"source": fpath, "filename": file}
            documents.append(Document(id=doc_id, content=text, metadata=meta))
    return documents

def load_files(paths: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield raw docs: {doc_id, title, source, text}."""
    for path in paths:
        for doc in load_docs(path):
            yield {
                "doc_id": doc.id,
                "title": doc.metadata.get("filename") if doc.metadata else "",
                "source": doc.metadata.get("source") if doc.metadata else "",
                "text": doc.content,
            }

def split_into_chunks(
    doc: Dict[str, Any],
    chunk_size: int = 800,
    chunk_overlap: int = 120
) -> Iterator[Dict[str, Any]]:
    """
    Yield chunks with whitespace-aware boundaries.

    Expects doc: {doc_id, title, source, text}
    Yields: {doc_id, chunk_id, content, text, title, source, ord}
    """
    doc_id = str(doc.get("doc_id", "")).strip() or "doc"
    title = str(doc.get("title", "") or "")
    source = str(doc.get("source", "") or "")
    text = str(doc.get("text", "") or "").strip()

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

    def _snap_boundary(s: int, e: int, max_look: int = 50) -> int:
        # Try forward to next whitespace/newline
        fwd = e
        limit = min(n, e + max_look)
        while fwd < limit and not text[fwd].isspace():
            fwd += 1
        if fwd < n and fwd != e:
            return fwd
        # Try backward to previous whitespace/newline
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
                "text": chunk_text,  # provide both keys for callers
                "title": title,
                "source": source,
                "ord": idx,
            }
            idx += 1
        # advance start using overlap
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end  # avoid infinite loop
        start = next_start

def chunk_docs(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter: str = "recursive"
) -> List[Chunk]:
    """Produce deterministic chunk_ids; return list of Chunk dataclasses."""
    chunks: List[Chunk] = []
    for doc in docs:
        doc_dict = {
            "doc_id": doc.id,
            "title": doc.metadata.get("filename") if doc.metadata else "",
            "source": doc.metadata.get("source") if doc.metadata else "",
            "text": doc.content,
        }
        # For now only one splitter implemented (simple paragraph/sentence aware)
        # You can add additional strategies later.
        parts = list(split_into_chunks(doc_dict, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        for c in parts:
            idx = c["ord"]
            # deterministic chunk uuid
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}::{idx}"))
            chunks.append(Chunk(id=cid, content=c["content"], doc_id=doc.id, chunk_index=idx, chunk_id=c["chunk_id"], metadata={"title": c["title"], "source": c["source"]}))
    return chunks