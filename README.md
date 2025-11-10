# Ask My Docs

Retrieval playground that indexes three curated dental PDFs and exposes BM25/vector
search through a Weaviate (default) or Pinecone backend. The legacy crawler +
normalization workflow has been removed; ingestion now works only with the static books
checked into `data/dental_corpus/`.

## Quickstart
1. **Start Weaviate locally**
   ```bash
   docker compose up -d
   ```
2. **Install Python deps**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Ingest the dental PDFs**
   ```bash
   python scripts/ingest.py --chunk-size 800 --chunk-overlap 120
   ```
   The script walks `data/dental_corpus/`, extracts text from each PDF, chunks the text,
   embeds every chunk, and upserts it into the configured collection. Every chunk stores
   a `book_name` property so it is obvious which PDF it originated from.
4. **Query the collection**
   ```bash
   python scripts/search.py --q "nerve block protocol" --mode hybrid --k 5
   ```
5. **(Optional) Visualize local embeddings**
   ```bash
   python scripts/viz.py --source local --sample-size 200
   ```
   Generates `artifacts/embeddings.png` by encoding the same three PDFs locally.

## Container image

Build a production-ready image (multi-stage, non-root) and inject build metadata with args:

```bash
docker build \
  --build-arg APP_BUILD_VERSION=$(git describe --always --dirty) \
  --build-arg APP_BUILD_SHA=$(git rev-parse --short HEAD) \
  --build-arg APP_BUILD_TIME=$(date -u +%FT%TZ) \
  -t ask-my-docs:latest .
```

Run it by providing configuration exclusively through environment variables (no secrets baked into the image):

```bash
docker run --rm -p 8000:8000 \
  -e WEAVIATE_URL=http://host.docker.internal:8080 \
  -e WEAVIATE_COLLECTION=dental-dev \
  -e OPENAI_API_KEY=sk-your-key \
  ask-my-docs:latest
```

Key endpoints exposed by `uvicorn src.server:app`:

- `GET /health` → returns `{"status": "ok", "build": {version, sha, time}}` for liveness checks.
- `POST /query` → accepts `{query, mode, k, alpha, collection}` and streams back retrieval results.

Example curl calls:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "pulpitis treatment", "mode": "bm25", "k": 5}'
```

The `.dockerignore` prevents local data, git history, and `.env` secrets from entering the build context. Provide everything the container needs (vector-store URLs, API keys, etc.) through env vars when running.
