from chromadb import Client

# llm/rag/retriever.py
class RAGRetriever:
    def __init__(self, top_k=4):
        client = get_chroma_client()
        self.col = get_collection(client)
        self.top_k = top_k

    def retrieve(self, query, topic=None):
        where = {"topic": topic} if topic else None
        res = self.col.query(
            query_texts=[query],
            n_results=self.top_k,
            where=where
        )
        return res["documents"][0] if res["documents"] else []