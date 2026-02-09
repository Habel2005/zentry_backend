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

async def handle_llm(call_id, caller_id, phone, text_ml) -> tuple:
    """
    Returns: (response_type, content, text_for_db)
    """
    start_time = time.time()
    print(f"\n--- 🗣️ DETECTED (ML): {text_ml} ---")

    # --- REFLEX 1: GREETING (Lat: <100ms) ---
    greetings = ["ഹലോ", "നമസ്കാരം", "hello", "hi"]
    cleaned = text_ml.lower().strip().replace("!", "")
    
    if cleaned in greetings or len(cleaned) < 3:
        # LOGGING: Even though it's a reflex, log it!
        log_message(call_id, "ai", "Hello! (Reflex)") 
        return ("reflex", "intro", "Hello! (Reflex)")

    # --- STEP 1: Translate (Lat: ~200ms) ---
    t0 = time.time()
    text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    lat_trans = int((time.time() - t0) * 1000)
    
    log_processing_step(call_id, "translate_ml_en", {"raw": text_ml}, {"trans": text_en}, latency_ms=lat_trans)

    # --- REFLEX 2: NOISE/SILENCE ---
    if not text_en or len(text_en) < 2:
        return ("reflex", "error", "[System: Audio Unclear]")

    # --- STEP 2: Intent (Lat: ~50ms) ---
    intent = await cpu_scheduler.run(detect_intent, text_en)
    log_intent(call_id, intent)

    # --- STEP 3: RAG (Lat: ~100ms) ---
    rag_topic = INTENT_TO_TOPIC.get(intent, None)
    rag_docs = []
    
    if rag_topic:
        rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, rag_topic)
    if not rag_docs:
        rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, None)

    # --- REFLEX 3: FALLBACK (Guardrail) ---
    # If specific query but NO data, don't waste 4s generating a hallucination.
    if not rag_docs and intent not in ["general", "eligibility"]:
        log_message(call_id, "ai", "I don't have that info. (Reflex)")
        return ("reflex", "fallback", "I don't have that info.")

    # --- STEP 4: LLM Generation (Lat: ~3000ms) ---
    t1 = time.time()
    snapshot = get_snapshot(caller_id, intent)
    
    # Fetch history safely
    history = []
    if session_store:
        session = session_store.get_session(phone)
        history = session.get("history", [])[-4:] # Keep context tight

    prompt = build_prompt(text_en, rag_docs, history, snapshot)
    response_en = await gpu_scheduler.run(engine.generate, prompt)
    
    # Guardrails
    # ... (Keep your existing guardrail logic) ...

    lat_llm = int((time.time() - t1) * 1000)
    log_processing_step(call_id, "llm_generate", {"prompt_len": len(prompt)}, {"response": response_en}, latency_ms=lat_llm)

    # --- STEP 5: Translate Back (Lat: ~200ms) ---
    reply_ml = await cpu_scheduler.run(en_to_ml, response_en)
    
    # Update Session
    if session_store:
        new_history = history + [{"role": "user", "text": text_en}, {"role": "ai", "text": response_en}]
        session_store.update_session(phone, {"history": new_history})
        session_store.persist_later(phone)

    # Final Database Log
    log_message(call_id, "ai", reply_ml)
    
    return ("text", reply_ml, reply_ml)

    