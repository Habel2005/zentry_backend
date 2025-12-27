# llm/brain.py
import asyncio
from llm.intent import detect_intent, detector as shared_detector
from llm.engine import PhiEngine
from llm.scheduler import gpu_scheduler, cpu_scheduler
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.rag.retriever import RAGRetriever
from llm.rag.embedder import Embedder
from llm.translate import ml_to_en, en_to_ml
from session.session_store import SessionStore

# Initialize Singletons
engine = PhiEngine("models/phi-4-mini-instruct.Q4_K_M.gguf")
rag = RAGRetriever()
global_embedder = Embedder()
# Assuming you initialize SessionStore with your creds in main_server.py and pass it here
# For now, we will create a global instance or pass it in handle_llm
# Ideally, pass the session_store instance from main_server.
session_store = None 

def init_globals(store_instance):
    global session_store
    session_store = store_instance

async def handle_llm(phone: str, text_ml: str) -> str:
    session = session_store.get_session(phone)
    history = session.get("history", [])[-6:]

# ---------------------------------------------------------
    # STEP 1: Translate (CPU Bound)
    # ---------------------------------------------------------
    # We use cpu_scheduler so we don't block the WebSocket loop
    text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    
    # ---------------------------------------------------------
    # STEP 2: Intent & RAG (CPU/IO Bound) -> Run in Parallel
    # ---------------------------------------------------------
    # Running these together saves time (e.g., 100ms + 100ms = 100ms total)
    intent_task = cpu_scheduler.run(detect_intent, text_en)
    rag_task = cpu_scheduler.run(rag.retrieve, text_en)
    
    intent, rag_docs = await asyncio.gather(intent_task, rag_task)

    # ---------------------------------------------------------
    # STEP 3: Build Prompt (Fast, no scheduler needed)
    # ---------------------------------------------------------
    prompt = build_prompt(text_en, rag_docs, history)

    # ---------------------------------------------------------
    # STEP 4: LLM Generation (GPU Bound)
    # ---------------------------------------------------------
    # We use gpu_scheduler here to protect VRAM
    response_en = await gpu_scheduler.run(engine.generate, prompt)

    # ---------------------------------------------------------
    # STEP 5: Guardrails & Translate Back (CPU Bound)
    # ---------------------------------------------------------
    safety_response = apply_guardrails(response_en, intent, rag_docs)
    final_en = safety_response if safety_response else response_en

    new_history = history + [
        {"role": "user", "text": text_en},
        {"role": "ai", "text": final_en}
    ]

    session_store.update_session(phone, {"history": new_history[-6:]})
    session_store.persist_later(phone)

    return en_to_ml(final_en)
