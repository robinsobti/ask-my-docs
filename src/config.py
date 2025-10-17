import os
from dotenv import load_dotenv

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

COLLECTION_NAME = "Docs"

# Schema definition for the Docs collection used in v1 (BM25 + client vectors).
DOCS_SCHEMA = {
    "name": COLLECTION_NAME,
    "description": "Text chunks from documents (BM25 enabled; vectors stored client-side)",
    "properties": [
        {"name": "stable_id", "dataType": "text", "description": "Stable chunk identifier doc_id::chunk_id"},
        {"name": "text", "dataType": "text", "description": "Chunk text"},
        {"name": "title", "dataType": "text", "description": "Document title"},
        {"name": "source", "dataType": "text", "description": "Source path/URL"},
        {"name": "doc_id", "dataType": "text", "description": "Stable document id"},
        {"name": "chunk_id", "dataType": "text", "description": "Stable chunk id"},
        {"name": "ord", "dataType": "int", "description": "Chunk order within the original document"},
    ],
    "vectorizer": "none",
    "moduleConfig": {
        "bm25": {"enabled": True}
    }
}
