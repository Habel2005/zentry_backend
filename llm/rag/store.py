# llm/rag/store.py
import chromadb # Use the main module to access PersistentClient

def get_chroma_client(persist_path="rag_db"):
    # Modern ChromaDB persistence
    return chromadb.PersistentClient(path=persist_path)

def get_collection(client, name="admission"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )