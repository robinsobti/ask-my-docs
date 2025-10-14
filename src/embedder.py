"""
Model choices (pick one):


sentence-transformers/all-MiniLM-L6-v2 (dim=384, fast)


bge-small-en-v1.5 (dim=384, strong recall for size)


Functions / class (signatures):

 class Embedder:
    def __init__(self, model_name: str, device: str | None = None,
                 batch_size: int = 64, normalize: bool = True):
        ...


    @property
    def dim(self) -> int: ...


    def encode(self, texts: list[str]) -> "ndarray[float]":
        
Notes:


Normalize vectors to unit length (cosine ≈ dot). Keep a toggle.


Batch encode for speed; keep memory sane.

"""
from sentence_transformers import SentenceTransformer
from numpy import ndarray
from typing import Optional
import torch
class Embedder:
    def __init__(self, model_name: str, device: Optional[str] = None,
                 batch_size: int = 64, normalize: bool = True):
        self.model = SentenceTransformer(model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.batch_size = batch_size
        self.normalize = normalize

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> ndarray:
        """Return (n, dim), normalized if configured."""
        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=self.normalize)
        return embeddings
    
if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    print(f"Model dim: {embedder.dim}")
    texts = ["Hello world", "How are you?"]
    vecs = embedder.encode(texts)
    print(vecs)
    print(vecs.shape)