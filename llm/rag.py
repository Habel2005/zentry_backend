
from chromadb import Client

class RAGRetriever:
    def __init__(self):
        self.client = Client()
        self.col = self.client.get_or_create_collection("admission")

    def retrieve(self, query, k=4):
        res = self.col.query(query_texts=[query], n_results=k)
        return res["documents"][0] if res["documents"] else []
