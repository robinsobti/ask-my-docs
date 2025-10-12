# src/embedder.py
from __future__ import annotations
import numpy as np
from typing import List, Optional

class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,       # e.g., "cuda", "mps", or "cpu"
        batch_size: int = 64,
        normalize: bool = True,
        dtype: str = "float32",             # keep vectors compact
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.dtype = np.float32 if dtype == "float32" else np.float64

        # Load model
        self.model = SentenceTransformer(model_name, device=device)
        # Infer dimension once
        _probe = self.model.encode(["probe"], normalize_embeddings=False)
        self._dim = int(_probe.shape[-1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 array, L2-normalized if configured."""
        if not texts:
            return np.zeros((0, self.dim), dtype=self.dtype)

        # SentenceTransformers already batches internally, but we can still pass batch_size
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=False,  # we’ll control normalization explicitly
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(self.dtype, copy=False)

        if self.normalize:
            # Stable L2 norm with epsilon to avoid div-by-zero on empty/whitespace strings
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            vecs /= norms

        return vecs

if __name__ == "__main__":
    emb = Embedder("BAAI/bge-small-en-v1.5", batch_size=64, normalize=True)
    xs = ["What is the refund policy?", "How do I get a refund?"]
    V = emb.encode(xs)
    print("dim:", emb.dim, "shape:", V.shape)          # (2, 384) for BGE-small
    # cosine similarity of the two sentences (should be relatively high)
    sim = float(V[0] @ V[1])
    print("cosine:", sim)