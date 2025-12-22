# llm/rag/ingest_qa.py
import os, re, uuid
from PyPDF2 import PdfReader
from llm.rag.embedder import Embedder
from llm.rag.store import get_chroma_client, get_collection

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_qa(text):
    q, a, out = None, [], []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            if q and a:
                out.append((q, clean(" ".join(a))))
            q, a = line[2:].strip(), []
        elif line.startswith("A:"):
            a.append(line[2:].strip())
        elif a:
            a.append(line)
    if q and a:
        out.append((q, clean(" ".join(a))))
    return out

def ingest_qa_pdf(path, source):
    client = get_chroma_client()
    col = get_collection(client)
    embedder = Embedder()

    text = "\n".join(
        p.extract_text() or ""
        for p in PdfReader(path).pages
    )

    docs, metas, ids = [], [], []
    for q, a in extract_qa(text):
        docs.append(f"Question: {q}\nAnswer: {a}")
        metas.append({"source": source, "type": "qa"})
        ids.append(str(uuid.uuid4()))

    embeddings = embedder.embed(docs)
    col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

    client.persist()
    print(f"✅ Ingested {len(docs)} QA pairs from {path}")
