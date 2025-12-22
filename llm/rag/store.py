from chromadb import Client
from chromadb.config import Settings

def get_chroma_client(persist_path="rag_db"):
    return Client(
        Settings(
            persist_directory=persist_path,
            anonymized_telemetry=False
        )
    )

def get_collection(client, name="admission"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
