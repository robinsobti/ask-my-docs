# Ask-My-Docs (v0)

**v0 goal:** runnable skeleton with Weaviate (local), BM25 keyword search, no embeddings.

## Quickstart
1. Start DB:
   ```bash
   docker compose up -d

Install deps:

 python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


Ingest sample docs:

 python scripts/ingest.py


Search:

 python scripts/search.py --q "refund packaging" --k 5


You should see a table of results with BM25 scores.
