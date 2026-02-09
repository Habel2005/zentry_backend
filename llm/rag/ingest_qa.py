# llm/rag/ingest_qa.py
import re, uuid
from llm.rag.embedder import embedder_instance # Use shared embedder
from llm.rag.store import get_chroma_client, get_collection
from llm.rag.loader import load_file

def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_qa(text):
    """
    Parses both 'Q: ... A: ...' and 'Q1. ... A1. ...' formats.
    """
    q, a, out = None, [], []
    
    # Regex to catch "Q1.", "Q10.", "Q.", "Question:"
    q_pattern = re.compile(r"^(Q\d*[\.:]|Question:)\s*(.*)", re.IGNORECASE)
    # Regex to catch "A1.", "A10.", "A.", "Answer:"
    a_pattern = re.compile(r"^(A\d*[\.:]|Answer:)\s*(.*)", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        if not line: continue

        q_match = q_pattern.match(line)
        a_match = a_pattern.match(line)

        if q_match:
            # If we were building an answer, save the previous pair
            if q and a:
                out.append((q, clean(" ".join(a))))
            # Start new question
            q = q_match.group(2).strip()
            a = []
        elif a_match:
            # Start new answer
            a.append(a_match.group(2).strip())
        elif a is not None:
            # Continue answer (multi-line)
            a.append(line)
    
    # Save last pair
    if q and a:
        out.append((q, clean(" ".join(a))))
    return out

def ingest_qa_file(path, source):
    client = get_chroma_client()
    col = get_collection(client)
    
    print(f"📄 Parsing QA File: {path}...")
    text = load_file(path) # Use the generic loader

    docs, metas, ids = [], [], []
    qa_pairs = extract_qa(text)

    if not qa_pairs:
        print(f"⚠️ No QA pairs found in {path}. Check formatting.")
        return

    for q, a in qa_pairs:
        # Format explicitly for the LLM
        docs.append(f"Question: {q}\nAnswer: {a}")
        metas.append({"source": source, "type": "qa", "topic": "faq"}) # Always tag as FAQ
        ids.append(str(uuid.uuid4()))

    print(f"🧠 Embedding {len(docs)} QA pairs...")
    embeddings = embedder_instance.embed(docs)
    
    col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"✅ Ingested {len(docs)} QA pairs from {path}")