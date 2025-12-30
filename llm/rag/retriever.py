# llm/rag/retriever.py
from llm.rag.store import get_chroma_client, get_collection

class RAGRetriever:
    def __init__(self, embedder_instance, top_k=3):
        """
        Args:
            embedder_instance: The shared Embedder object from embedder.py
        """
        # Use the helper from store.py for consistency
        client = get_chroma_client() 
        self.col = get_collection(client, name="admission")
        
        self.embedder = embedder_instance 
        self.top_k = top_k

    def retrieve(self, query, topic=None):
        # 1. Embed query using the shared model
        query_vector = self.embedder.embed([query]) 

        # 2. Build Filter
        where = {"topic": topic} if topic else None

        # 3. Query using EMBEDDINGS
        res = self.col.query(
            query_embeddings=query_vector, 
            n_results=self.top_k,
            where=where
        )
        
        return res["documents"][0] if res["documents"] else []