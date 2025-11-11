# Ask My Docs

Ask My Docs ingests three curated dental PDFs, chunks them, and pushes the chunks to a
vector store (Pinecone by default, Weaviate optional). It includes:

- **Ingestion CLI (`scripts/ingest.py`)** – loads PDFs under `data/dental_corpus/`, drops
  front-matter pages, chunks text, attaches `book_name` metadata, and upserts vectors.
- **Search CLI (`scripts/search.py`)** – vector-only query interface with optional
  reranking and trace logging.
- **Streamlit demo (`scripts/demo_app.py`)** – "Ask My Docs" UI with Dental/E-Commerce
  tabs, sample queries, Pinecone badge, and an answer provenance indicator.
- **Local visualization (`scripts/viz.py`)** – encodes PDFs locally and projects
  embeddings for sanity checks.

The crawler + normalization pipeline has been removed; the repo is intentionally small
and focused on the static dental corpus.

## Prerequisites
- Python 3.11+
- OpenAI API key (for embeddings/generation)
- Pinecone project (default) or a local Weaviate instance (`docker compose up -d`)

Set the usual environment variables (`OPENAI_API_KEY`, `PINECONE_API_KEY`,
`PINECONE_ENVIRONMENT`, optional `WEAVIATE_URL`).

## Setup & ingestion
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional: start Weaviate instead of Pinecone
docker compose up -d

# ingest the bundled PDFs (tweak chunk params to taste)
python scripts/ingest.py data/dental_corpus \
  --collection dental \
  --chunk-size 450 \
  --chunk-overlap 90 \
  --model text-embedding-3-small
```

`scripts/ingest.py` logs each document, chunk count, and upsert summary.

## Querying & demos
- **Vector search (CLI)**
  ```bash
  python scripts/search.py \
    --q "How does dental enamel protect the tooth crown from wear?" \
    --collection dental \
    --mode vector \
    --k 5
  ```
- **Streamlit UI**
  ```bash
  streamlit run scripts/demo_app.py
  ```
  Use the Dental tab's sample queries to auto-fill the textbox; the answer card shows
  `Answering from: <collection>` so provenance stays clear.
- **Embedding visualization**
  ```bash
  python scripts/viz.py --source local --sample-size 200
  ```
  Saves `artifacts/embeddings.png` with PCA/UMAP/t-SNE projections.

## Testing
The suite uses `pytest`.
```bash
pytest
```
Some tests (e.g., integration against Weaviate) require the vector store and OpenAI
credentials; skip or mock those when necessary.

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
- `POST /query` → accepts `{query, mode, k, alpha, collection}` (use `mode=vector` when Pinecone is configured) and streams back retrieval results.

Example curl calls:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "pulpitis treatment", "mode": "vector", "k": 5}'
```

The `.dockerignore` prevents local data, git history, and `.env` secrets from entering the build context. Provide everything the container needs (vector-store URLs, API keys, etc.) through env vars when running.
