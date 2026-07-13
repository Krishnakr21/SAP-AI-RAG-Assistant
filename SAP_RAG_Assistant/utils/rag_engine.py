# utils/rag_engine.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RAGEngine:

    def __init__(self):
        print("Loading embedding model... (this takes a moment)")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load FAISS index if exists, else create new
        self.index = None

    def build_index(self, documents):
        print("Building FAISS index...")

        embeddings = self.embedding_model.encode(documents)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print("Index built successfully!")

    def search(self, query, documents, top_k=2):
        """Search for the most relevant SAP documents."""

        query_vec = self.embedding_model.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:

            # ✔ FIX: Skip invalid FAISS indexes
            if 0 <= idx < len(documents):
                results.append(documents[idx])

        if not results:
            return ["No relevant document found."]

        return results

