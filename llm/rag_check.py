# check_db.py
from llm.rag.store import get_chroma_client, get_collection

def check():
    print("🕵️ Checking Brain Memory...")
    client = get_chroma_client()
    col = get_collection(client)
    
    count = col.count()
    print(f"🧠 Total Memories: {count}")
    
    if count == 0:
        print("❌ BRAIN IS EMPTY! Please run 'python -m llm.seed_brain' again.")
    else:
        print("✅ Brain is healthy.")
        # Test a query
        results = col.query(query_texts=["fees"], n_results=1)
        print("Test Query 'fees':", results['documents'][0][0][:50])

if __name__ == "__main__":
    check()