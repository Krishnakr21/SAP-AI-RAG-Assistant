import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# File paths
SAP_DATA_FILE = os.path.join(DATA_DIR, "sap_documents.csv")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")

# Embedding model name
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
