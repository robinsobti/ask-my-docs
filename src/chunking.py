from typing import List
from dataclasses import dataclass
import os
import uuid
import re
from PyPDF2 import PdfReader

@dataclass
class Document:
    id: str
    content: str
    metadata: dict  = None

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
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.txt') or file.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc_id = str(uuid.uuid4())
                documents.append(Document(id=doc_id, content=content, metadata={'source': file_path}))
            elif file.endswith('.pdf'):
                try:
                    reader = PdfReader(file_path)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                    doc_id = str(uuid.uuid4())
                    documents.append(Document(id=doc_id, content=content, metadata={'source': file_path}))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return documents

def chunk_docs(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    splitter: str = "recursive"
) -> List[Chunk]:
    """Produce deterministic chunk_ids; return list of chunks."""
    chunks = []
    for doc in docs:
        content = doc.content
        if splitter == "by_paragraph":
            parts = re.split(r'\n\s*\n', content)
        elif splitter == "by_sentence":
            parts = re.split(r'(?<=[.!?]) +', content)
        else:  # recursive or default
            parts = re.split(r'(?<=[.!?]) +|\n\s*\n', content)

        current_chunk = ""
        chunk_index = 0
        for part in parts:
            if len(current_chunk) + len(part) + 1 <= chunk_size:
                current_chunk += (part + " ")
            else:
                if current_chunk:
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}::{chunk_index}"))
                    chunks.append(Chunk(id=chunk_id, content=current_chunk.strip(), doc_id=doc.id, chunk_index=chunk_index, metadata=doc.metadata))
                    chunk_index += 1
                current_chunk = part + " "
                while len(current_chunk) > chunk_size:
                    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}::{chunk_index}"))
                    chunks.append(Chunk(id=chunk_id, content=current_chunk[:chunk_size].strip(), doc_id=doc.id, chunk_index=chunk_index, metadata=doc.metadata))
                    chunk_index += 1
                    current_chunk = current_chunk[chunk_size - chunk_overlap:]

        if current_chunk:
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc.id}::{chunk_index}"))
            chunks.append(Chunk(id=chunk_id, content=current_chunk.strip(), doc_id=doc.id, chunk_index=chunk_index, metadata=doc.metadata))

    return chunks