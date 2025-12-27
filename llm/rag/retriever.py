# llm/rag/retriever.py
from chromadb import Client, PersistentClient

class RAGRetriever:
    def __init__(self, embedder_instance, top_k=3):
        """
        Args:
            embedder_instance: The shared Embedder object from brain.py
        """
        # Connect to DB
        client = PersistentClient(path="rag_db")
        self.col = client.get_or_create_collection("admission")
        
        # Store the shared model reference
        self.embedder = embedder_instance 
        self.top_k = top_k

    def retrieve(self, query, topic=None):
        # 1. Use YOUR model to convert text -> numbers
        # This uses the already loaded model in RAM (Fast!)
        query_vector = self.embedder.embed([query]) 

        # 2. Build Filter
        where = {"topic": topic} if topic else None

        # 3. Query using EMBEDDINGS, not TEXTS
        # This stops Chroma from loading its own model.
        res = self.col.query(
            query_embeddings=query_vector, 
            n_results=self.top_k,
            where=where
        )
        
        return res["documents"][0] if res["documents"] else []