from __future__ import annotations
from pathlib import Path              # built-in Python module, no install needed
from dataclasses import dataclass
from pypdf import PdfReader
import uuid

NAMESPACE_DOCUMENT = uuid.uuid4()

@dataclass
class Document:
    title: str
    text: str

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title: str
    source: str | None = None

def load_docs(path: str) -> list[Document]:
    """Walk a folder; read .txt/.md directly; extract text from .pdf (basic)."""
    docs = []
    p = Path(path)
    for file in p.rglob("*"):
        if file.suffix in [".txt", ".md"]:
            with open(file, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(Document(title=file.stem, text=text))
        elif file.suffix == ".pdf":
            text = extract_text_from_pdf(file)
            docs.append(Document(title=file.stem, text=text))
    return docs

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using PyPDF."""
    text = ""
    try:
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return text


def chunk_docs(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter: str = "recursive"
) -> list[Chunk]:
    """Produce deterministic chunk_ids; return list of chunks.

    This is a simple fixed-window chunker: it slices text into windows of
    `chunk_size` characters with `chunk_overlap` overlap. `doc_id` is set to
    the document title by default.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        text = doc.text or ""
        start = 0
        idx = 0
        text_len = len(text)
        if text_len == 0:
            continue
        while start < text_len:
            chunk_uuid = uuid.uuid5(NAMESPACE_DOCUMENT, f"{doc.title}#{idx}")
            end = start + chunk_size
            piece = text[start:end]
            chunks.append(Chunk(doc_id=doc.title, chunk_id=chunk_uuid, text=piece, title=doc.title))
            idx += 1
            # advance with overlap
            if end >= text_len:
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks
