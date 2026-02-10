# seed_brain.py
import os
import shutil
from llm.rag.ingest_docs import ingest_document
from llm.rag.ingest_qa import ingest_qa_file

# Define your files here
FILES = [
    # (Filename, Source Label, Default Topic)
    ("rag_training_data.pdf", "official_brochure", "general"), 
    ("rag Q&A.docx", "counselling_faq", "faq")
]

def train():
    print("🧠 Starting Brain Training...")

    # 1. Reset Database (Optional: safer for testing)
    if os.path.exists("rag_db"):
        shutil.rmtree("rag_db")
        print("🗑️  Old memory wiped.")

    # 2. Ingest
    for filename, source, topic in FILES:
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue

        if "Q&A" in filename:
            ingest_qa_file(filename, source)
        else:
            # We tag the PDF as 'general' because it has everything.
            # The new brain.py logic will handle finding 'fees' inside it.
            ingest_document(filename, source, topic)

if __name__ == "__main__":
    train()