# src/embedder.py
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import torch
except Exception as e:  # pragma: no cover
    SentenceTransformer = None
    torch = None
    _IMPORT_ERROR = e

class Embedder:
    """
    Thin wrapper around Sentence-Transformers.

    - Loads a model once; reuses it across calls.
    - Returns np.ndarray of dtype float32.
    - Supports optional L2 normalization.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: Optional[str] = None,
        batch_size: int = 32,
        dtype: np.dtype = np.float32,
    ) -> None:
        """
        Args:
            model_name: HF model id for sentence-transformers.
            device: 'cuda', 'mps', 'cpu', or None to auto-detect.
            batch_size: encode batch size.
            dtype: output dtype (defaults to float32).
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for Embedder. "
                "Install with: pip install sentence-transformers\n"
            ) from _IMPORT_ERROR

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.dtype = dtype

        if device is None and torch is not None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # Apple Silicon
                device = "mps"
            else:
                device = "cpu"
        elif device is None:
            device = "cpu"

        self.device = device
        self._model = SentenceTransformer(model_name, device=device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts: list of strings to embed.
            normalize: if True, returns L2-normalized vectors.

        Returns:
            np.ndarray with shape (len(texts), dim) and dtype float32 (by default).
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a List[str].")
        if len(texts) == 0:
            return np.empty((0, self._dim), dtype=self.dtype)

        # SentenceTransformer can normalize for us efficiently.
        # convert_to_numpy=True returns float32 by default.
        vecs: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            device=self.device,
            show_progress_bar=False,
        )

        # Ensure dtype
        if vecs.dtype != self.dtype:
            vecs = vecs.astype(self.dtype, copy=False)

        # If user disabled normalize, optionally enforce dtype and shape
        if not normalize:
            # (Optional) L2-safe guard against all-zeros (shouldn't happen with ST)
            # but no-op here; user explicitly asked for non-normalized.
            pass

        return vecs

    @property
    def dim(self) -> int:
        """Embedding dimensionality for the loaded model."""
        return self._dim
    
def main():
    embedder: Embedder = Embedder()
    print(f'Dimensions are {embedder.dim}')
    print(f"Vectors: {embedder.encode(['hey, how are you?', 'I\'m doing well buddy'])}")

if __name__ == "__main__":
    main()