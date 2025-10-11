import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

COLLECTION_NAME = "Docs"

# Schema for v0 (BM25-only; vectorizer=None)
DOCS_SCHEMA = {
    "name": COLLECTION_NAME,
    "description": "Text chunks from documents (BM25 only in v0)",
    "properties": [
        {"name": "text", "dataType": "text", "description": "Chunk text"},
        {"name": "title", "dataType": "text", "description": "Document title"},
        {"name": "source", "dataType": "text", "description": "Source path/URL"},
        {"name": "doc_id", "dataType": "text", "description": "Stable document id"},
        {"name": "chunk_id", "dataType": "text", "description": "Stable chunk id"},
    ],
    "vectorizer": "none",      # important for v0
    "moduleConfig": {
        "bm25": {"enabled": True}
    }
}