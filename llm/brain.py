# llm/brain.py
import asyncio
import logging
from pydoc import text
import time
from llm.intent import detect_intent, detector as shared_detector
from llm.engine import PhiEngine
from llm.scheduler import gpu_scheduler, cpu_scheduler
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.rag.retriever import RAGRetriever
from llm.rag.embedder import embedder_instance 
from llm.translate import ml_to_en, en_to_ml
from db.call_repo import log_message
from db.ai_repo import log_processing_step, log_intent
from db.snapshot_repo import get_snapshot
from zentry_backend.db.client import init_supabase

# Initialize Singletons
engine = PhiEngine("models/llm/Phi-4-mini-instruct-Q5_K_M.gguf")
rag = RAGRetriever(embedder_instance=embedder_instance)
session_store = None 

INTENT_TO_TOPIC = {
    "seat": "seats",
    "fee": "fees",
    "placement": "placements",
    "eligibility": "requirements"
}

def init_globals(store_instance):
    global session_store
    session_store = store_instance

# llm/brain.py
# ... (Imports remain same) ...

async def handle_llm(call_id, caller_id, phone, text_ml) -> tuple:
    print(f"\n--- 🗣️ DETECTED (ML): {text_ml} ---")

    # --- 1. REFLEX CHECK ---
    greetings = ["ഹലോ", "നമസ്കാരം", "hello", "hi"]
    cleaned = text_ml.lower().strip().replace("!", "")
    if cleaned in greetings or len(cleaned) < 3:
        log_message(call_id, "ai", "Hello! (Reflex)") 
        return ("reflex", "intro", "Hello! (Reflex)")

    # --- 2. TRANSLATION ---
    t0 = time.time()
    try:
        text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    except Exception as e:
        print(f"❌ Translation Failed: {e}")
        text_en = text_ml # Fallback to raw text
    
    # 🔍 DEBUG: See what the brain actually hears
    print(f"🔍 DEBUG [Trans]: '{text_en}'") 
    
    log_processing_step(call_id, "translate_ml_en", text_ml, text_en, latency_ms=int((time.time()-t0)*1000))

    if not text_en or len(text_en) < 2:
        return ("reflex", "error", "Audio Unclear")

    # --- 3. INTENT ---
    intent = await cpu_scheduler.run(detect_intent, text_en)
    print(f"🔍 DEBUG [Intent]: {intent}")
    log_intent(call_id, intent)
    
    if intent in ["seat", "fee", "admission"]:
        init_supabase().table("interest_signals").insert({
            "call_id": call_id, "caller_id": caller_id, "quota_type": "general", "strength": "HIGH"
        }).execute()

    # --- 4. RAG SEARCH ---
    rag_topic = INTENT_TO_TOPIC.get(intent, None)
    rag_docs = []
    
    if rag_topic:
        print(f"🔍 DEBUG [RAG]: Searching topic '{rag_topic}'...")
        rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, rag_topic)
    
    if not rag_docs:
        print(f"🔍 DEBUG [RAG]: Fallback global search...")
        rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, None)

    # 🔍 DEBUG: Print found docs to console
    if rag_docs:
        print(f"🔍 DEBUG [Docs Found]: {len(rag_docs)}")
        print(f"📄 First Doc Snippet: {rag_docs[0][:100]}...")
    else:
        print("❌ DEBUG [Docs]: NO DOCUMENTS FOUND.")

    # GUARDRAIL: If specific intent but no docs, fail early
    if not rag_docs and intent not in ["general", "eligibility"]:
        print("🛡️ GUARDRAIL: No data -> Playing Fallback.")
        log_message(call_id, "ai", "I don't have that info. (Reflex)")
        return ("reflex", "fallback", "I don't have that info.")

    # --- 5. LLM GENERATION ---
    snapshot = get_snapshot(caller_id, intent)
    history = []
    if session_store:
        session = session_store.get_session(phone)
        history = session.get("history", [])[-4:]

    prompt = build_prompt(text_en, rag_docs, history, snapshot)
    
    # 🔍 DEBUG: Check what we are feeding the LLM
    print(f"🔍 DEBUG [Prompt]:\n{prompt[-300:]}") # Print last 300 chars of prompt

    response_en = await gpu_scheduler.run(engine.generate, prompt)
    print(f"🔍 DEBUG [LLM Raw]: {response_en}")

    # Guardrails
    safety_response = apply_guardrails(response_en, intent, rag_docs, shared_detector=None)
    final_en = safety_response if safety_response else response_en

    # --- 6. TRANSLATE BACK ---
    reply_ml = await cpu_scheduler.run(en_to_ml, final_en)
    print(f"🔍 DEBUG [Reply ML]: {reply_ml}")
    
    if session_store:
        new_history = history + [{"role": "user", "text": text_en}, {"role": "ai", "text": final_en}]
        session_store.update_session(phone, {"history": new_history})
        session_store.persist_later(phone)

    log_message(call_id, "ai", reply_ml)
    
    return ("text", reply_ml, reply_ml)

    