# llm/rag/embeddor.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        print(f"Loading Embedding Model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts):
        # Handle single string input just in case
        if isinstance(texts, str):
            texts = [texts]
            
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

# --- SINGLETON INSTANCE ---
# This runs once when you first import 'embedder_instance' anywhere
embedder_instance = Embedder()