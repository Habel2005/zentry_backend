# llm/rag/retriever.py
from llm.rag.store import get_chroma_client, get_collection

class RAGRetriever:
    def __init__(self, embedder_instance, top_k=6):
        client = get_chroma_client() 
        self.col = get_collection(client, name="admission")
        self.embedder = embedder_instance 
        self.top_k = top_k

    def retrieve(self, query, topic=None):
        query_vector = self.embedder.embed([query]) 
        where = {"topic": topic} if topic else None

        res = self.col.query(
            query_embeddings=query_vector, 
            n_results=self.top_k,
            where=where
        )
        
        # [IMPROVED] Return a list of dictionaries containing text and metadata
        results = []
        if res["documents"] and res["documents"][0]:
            for i in range(len(res["documents"][0])):
                results.append({
                    "text": res["documents"][0][i],
                    "source": res["metadatas"][0][i].get("source", "unknown"),
                    "topic": res["metadatas"][0][i].get("topic", "none")
                })
        return results