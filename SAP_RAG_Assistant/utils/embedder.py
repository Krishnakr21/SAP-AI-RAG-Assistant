from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class Embedder:
    def __init__(self):
        print("Loading embedding model... (this takes a moment)")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts):
        """Generate embeddings for a list of texts."""
        return self.model.encode(texts, convert_to_numpy=True)
