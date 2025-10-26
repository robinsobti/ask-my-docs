"""OpenAI embedding wrapper (implementation guidance only).

This module intentionally contains skeletons with step-by-step instructions so
you can wire the OpenAI embedding API yourself. Follow the notes in each
function to complete the implementation when you're ready.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



import argparse
from typing import Iterator, List

import numpy as np

from src.config import (
    DEFAULT_EMBEDDER_BATCH_SIZE,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_EMBED_DIM,
    OPENAI_API_KEY,
)


class Embedder:
    """Thin wrapper around OpenAI's embeddings endpoint (partially implemented)."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        *,
        batch_size: int = DEFAULT_EMBEDDER_BATCH_SIZE,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Prepare the embedder.

        Implementation plan:
        1. Validate that `OPENAI_API_KEY` is non-empty; raise a helpful error if it isn't.
        2. Store `model_name`, `batch_size`, and `dtype` on the instance (coerce batch_size to `int`).
        3. Instantiate an OpenAI client (e.g., `from openai import OpenAI`) and keep it on `self._client`.
        4. Set `self._dim` to the embedding width you intend to use (512 for `text-embedding-3-small`).
        5. Optionally prime any request kwargs you plan to reuse (like `dimensions=self._dim`).
        """

        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set to use the OpenAI embedder.")

        self.model_name = model_name
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.dtype = dtype
        self._dim = int(DEFAULT_EMBED_DIM)
        self._client = None  # Lazy client initialisation via _get_client
        self._request_kwargs = {"model": self.model_name, "dimensions": self._dim}

    def _get_client(self):
        """Return a lazily-instantiated OpenAI client.

        Implementation plan:
        1. If you've already created a client in `__init__`, simply return it here.
        2. Otherwise, instantiate the client on first use (again validating the API key) and cache it.
        3. Consider wrapping client creation in a try/except to surface credential errors clearly.
        """

        if getattr(self, "_client", None) is None:
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY must be set before creating the OpenAI client.")
            try:
                from openai import OpenAI  # Local import to avoid hard dependency during module import
            except ImportError as exc:  # pragma: no cover - library import guard
                raise ImportError("The 'openai' package is required. Install with 'pip install openai'.") from exc
            self._client = OpenAI(api_key=OPENAI_API_KEY)
        return self._client

    def _batched(self, texts: List[str]) -> Iterator[List[str]]:
        """Yield chunks of input texts respecting the configured batch size.

        Implementation plan:
        1. Iterate over `texts` in slices of length `self.batch_size`.
        2. Yield each slice while preserving order.
        3. Handle edge cases gracefully (empty list ⇒ no batches).
        """

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        total = len(texts)
        for start in range(0, total, self.batch_size):
            yield texts[start : start + self.batch_size]

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Embed a list of texts via OpenAI.

        Implementation plan:
        1. Validate that `texts` is a `List[str]`; return an empty array with shape `(0, self._dim)` when empty.
        2. Iterate over `_batched(texts)` to call the OpenAI embeddings endpoint batch-by-batch.
           - Use `self._get_client().embeddings.create(model=self.model_name, input=batch, dimensions=self._dim)`.
        3. Collect the `data[*].embedding` vectors from each response, preserving order.
        4. Stack the vectors into a NumPy array (`np.asarray(..., dtype=self.dtype)`).
        5. If `normalize` is True, L2-normalize each vector (see `_l2_normalize`).
        6. Return the resulting array.
        7. Add defensive error handling/logging as needed for API errors and rate limiting.
        """

        if not texts:
            return np.empty((0, self._dim), dtype=self.dtype)
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a List[str].")

        client = self._get_client()
        vectors: List[List[float]] = []
        request_kwargs = dict(self._request_kwargs)

        for batch in self._batched(texts):
            response = client.embeddings.create(input=batch, **request_kwargs)
            data = getattr(response, "data", None)
            if not data:
                raise RuntimeError(f"OpenAI embeddings response missing data (model={self.model_name}).")
            for item in data:
                embedding = getattr(item, "embedding", None)
                if embedding is None:
                    raise RuntimeError("Embedding item missing 'embedding' vector.")
                vectors.append(list(embedding))

        embeddings_np = np.asarray(vectors, dtype=self.dtype)
        if normalize:
            embeddings_np = self._l2_normalize(embeddings_np)
        return embeddings_np


    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Utility for L2-normalizing embeddings.

        Implementation plan:
        1. Guard against empty matrices (return the input unchanged).
        2. Compute row-wise norms (`np.linalg.norm(..., axis=1, keepdims=True)`).
        3. Avoid division by zero by clamping norms to a minimum epsilon.
        4. Divide each row by its norm and return the normalized matrix.
        """

        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return matrix / norms

    @property
    def dim(self) -> int:
        """Embedding dimensionality exposed to callers."""

        return getattr(self, "_dim", DEFAULT_EMBED_DIM)


def _parse_cli_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for quick embedding smoke tests."""

    parser = argparse.ArgumentParser(description="Embed sample texts using the OpenAI embedder.")
    parser.add_argument("texts", nargs="+", help="One or more strings to embed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDER_BATCH_SIZE,
        help=f"Batch size for embedding requests (default: {DEFAULT_EMBEDDER_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of the resulting vectors.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Run a simple end-to-end embedding using the configured OpenAI model."""

    # args = _parse_cli_args(argv)
    texts = ["hello world", "another example"]
    embedder = Embedder(batch_size=32)
    vectors = embedder.encode(texts, normalize=True)
    print(f"Embedded {len(texts)} texts with shape {vectors.shape} (normalize=True).")
    preview_rows = min(len(vectors), 2)
    if preview_rows:
        print("Preview (first rows):")
        for idx, row in enumerate(vectors[:preview_rows]):
            print(f"  [{idx}] {row[:6]} ...")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual smoke test entry point
    raise SystemExit(main())
