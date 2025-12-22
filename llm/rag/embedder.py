from sentence_transformers import SentenceTransformer

# Industry standard for RAG
MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
