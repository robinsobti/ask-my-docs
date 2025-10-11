import sys
from pathlib import Path
import weaviate
# Ensure project root is on sys.path so `from src...` imports work when
# running this script directly (python scripts/ingest.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.weaviate_store import search_bm25

def search_bm25_func(query, k):
    print(f"Top {k} results for query '{query}':")
    results = search_bm25(query, k)
    for i, r in enumerate(results):
        print(f"{i+1}. (score: {r['score']}) {r['title']} - {r['text'][:50]}... (source: {r['source']})")

if __name__ == "__main__":
    search_bm25_func('warranty', 3)