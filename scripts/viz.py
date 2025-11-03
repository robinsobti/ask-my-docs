#!/usr/bin/env python3
"""
Embedding visualization utilities.

Generates a 2-D projection (PCA/UMAP/t-SNE) of document chunk embeddings sourced
from either the live vector store (Weaviate) or locally encoded sample docs.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import load_files, split_into_chunks
from src.config import (
    COLLECTION_NAME,
    DEFAULT_EMBEDDER_MODEL,
    VECTOR_STORE_PROVIDER,
)
from src.embedder import Embedder

try:
    from src.vector_store.weaviate_store import create_collection_if_missing, get_client
except ImportError:  # pragma: no cover - optional dependency
    create_collection_if_missing = None
    get_client = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize embeddings as a 2-D scatter plot.")
    parser.add_argument(
        "--source",
        choices=["weaviate", "local"],
        default="weaviate" if VECTOR_STORE_PROVIDER == "weaviate" else "local",
        help="Where to pull embeddings from.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Maximum number of points to include in the plot.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "embeddings.png"),
        help="Path to save the visualization PNG.",
    )
    parser.add_argument(
        "--method",
        choices=["pca", "umap", "tsne"],
        default="pca",
        help="Dimensionality reduction technique (default: pca).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser


def _fetch_from_weaviate(sample_size: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    if get_client is None:
        raise RuntimeError("Weaviate client is unavailable; install requirements or use --source local.")

    client = get_client()
    create_collection_if_missing()
    response = client.collections.get(COLLECTION_NAME).query.fetch_objects(limit=sample_size)

    vectors: List[np.ndarray] = []
    labels: List[str] = []
    rng = random.Random(seed)

    for obj in response.objects:
        vector = np.asarray(obj.vector, dtype=np.float32)
        vectors.append(vector)
        doc_id = obj.properties.get("doc_id", "")
        title = obj.properties.get("title", "")
        chunk_id = obj.properties.get("chunk_id", "")
        label = f"{doc_id}::{chunk_id or obj.id}"
        if title:
            label += f" ({title})"
        labels.append(label)

    # Randomly sample if too many objects returned.
    if len(vectors) > sample_size:
        indices = list(range(len(vectors)))
        rng.shuffle(indices)
        indices = indices[:sample_size]
        vectors = [vectors[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    return np.array(vectors), labels


def _fetch_local(sample_size: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    data_dir = PROJECT_ROOT / "data" / "sample_docs"
    embedder = Embedder(model_name=DEFAULT_EMBEDDER_MODEL)

    chunks: List[Tuple[str, str]] = []
    for doc in load_files([str(data_dir)]):
        for chunk in split_into_chunks(doc):
            label = f"{chunk['doc_id']}::{chunk['chunk_id']} ({chunk.get('title') or ''})"
            chunks.append((chunk["text"], label))

    if not chunks:
        raise RuntimeError(f"No chunks produced from {data_dir}.")

    rng = random.Random(seed)
    rng.shuffle(chunks)
    chunks = chunks[:sample_size]

    texts = [text for text, _ in chunks]
    labels = [label for _, label in chunks]
    vectors = embedder.encode(texts, normalize=True)
    return np.array(vectors), labels


def _reduce_vectors(vectors: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        reducer = PCA(n_components=2, random_state=seed)
        return reducer.fit_transform(vectors)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("UMAP is not installed. Install umap-learn or use --method pca.") from exc
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(vectors)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:  # pragma: no cover - adheres to sklearn
            raise RuntimeError("scikit-learn is required for t-SNE.") from exc
        reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        return reducer.fit_transform(vectors)

    raise ValueError(f"Unknown reduction method '{method}'.")


def _plot(points: np.ndarray, labels: List[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], s=20, alpha=0.7)
    for (x, y), label in zip(points, labels):
        plt.text(x, y, label, fontsize=6, alpha=0.8)

    plt.title("Embedding Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.source == "weaviate":
            vectors, labels = _fetch_from_weaviate(args.sample_size, args.seed)
        else:
            vectors, labels = _fetch_local(args.sample_size, args.seed)

        if len(vectors) < 2:
            raise RuntimeError("Need at least two vectors to project.")

        reduced = _reduce_vectors(vectors, args.method, args.seed)
        _plot(reduced, labels, Path(args.output))

        print(f"Saved embedding visualization to {args.output}")
        return 0
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"Visualization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
