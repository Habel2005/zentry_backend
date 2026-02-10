# llm/rag/ingest_docs.py
import uuid
from llm.rag.loader import load_file
from llm.rag.chunker import chunk_text
from llm.rag.embedder import Embedder
from llm.rag.store import get_chroma_client, get_collection

def ingest_document(path, source, topic):
    client = get_chroma_client()
    col = get_collection(client)
    embedder = Embedder()

    text = load_file(path)
    chunks = chunk_text(text)
    embeddings = embedder.embed(chunks)

    col.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {
                "source": source,
                "topic": topic,
                "type": "doc"
            }
            for _ in chunks
        ]
    )

    # REMOVED: client.persist() - ChromaDB now handles this automatically
    print(f"✅ Ingested {len(chunks)} chunks from {path}")