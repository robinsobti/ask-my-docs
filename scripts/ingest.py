import sys
import uuid
from pathlib import Path

# Ensure project root is on sys.path so `from src...` imports work when
# running this script directly (python scripts/ingest.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weaviate_store import close_client, upsert_batch, upsert_batch_with_vectors
from typing import Iterator, Dict
from src.chunking import split_into_chunks
from src.embedder import Embedder
from src.config import COLLECTION_NAME

SAMPLE_DOCS = [
    {
        "title": "Warranty Policy",
        "source": "data/samples/warranty.txt",
        "text": "Our warranty covers defects in materials and workmanship for one year from the date of purchase.",
        "doc_id": "warranty",
        "chunk_id": "warranty#0",
    },
    {
        "title": "Returns & Refunds",
        "source": "data/samples/returns.txt",
        "text": "You may return items within 30 days for a full refund. Items must be unused and in original packaging.",
        "doc_id": "returns",
        "chunk_id": "returns#0",
    },
    {
        "title": "Shipping",
        "source": "data/samples/shipping.txt",
        "text": "We offer standard and express shipping options. Tracking numbers are provided for all shipments.",
        "doc_id": "shipping",
        "chunk_id": "shipping#0",
    },
]

def main():
    objs = []
    for d in SAMPLE_DOCS:
        objs.append({
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{d['doc_id']}::{d['chunk_id']}")),  # deterministic
            **d
        })
    n = upsert_batch(objs)
    print(f"Ingested {n} chunks.")

def upsert_with_vectors() -> None:
     """Load dir → chunk → embed (store vector in payload) → upsert."""
     embedder = Embedder()
     total = 0
     for d in SAMPLE_DOCS:
        doc = {
            "doc_id": d["doc_id"],
            "title": d["title"],
            "source": d["source"],
            "text": d["text"],
        }
        chunks = list(split_into_chunks(doc = doc))
        if not chunks:
            continue
        texts = [chunk["content"] for chunk in chunks]
        vectors = embedder.encode(texts)

        objs = []
        for c, vec in zip(chunks, vectors):
            objs.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c['doc_id']}::{c['chunk_id']}")),
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "title": c["title"],
                "source": c["source"],
                "text": c["text"],
            })

        # Use the vector-aware upsert (expects collection, list of objs, and vectors)
        n = upsert_batch_with_vectors(COLLECTION_NAME, objs, vectors)
        total += n

if __name__ == "__main__":
    try:
        upsert_with_vectors()
    finally:
        close_client()
