# llm/brain.py
import asyncio
import logging
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

async def handle_llm(call_id, caller_id, phone, text_ml) -> str:
    global session_store
    print(f"\n--- 🗣️ DETECTED SPEECH (ML): {text_ml} ---")
    
    # Safe History
    if session_store:
        session = session_store.get_session(phone)
        history = session.get("history", [])[-6:]
    else:
        history = []

    # STEP 1: Translate to English
    text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    print(f"--- 🇺🇸 TRANSLATED (EN): {text_en} ---")
    log_processing_step(call_id, "translate_ml_en", text_ml, text_en)

    # STEP 2: Intent Detection
    intent = await cpu_scheduler.run(detect_intent, text_en)
    print(f"--- 🎯 INTENT: {intent.upper()} ---")
    log_intent(call_id, intent)

    # STEP 3: RAG Retrieval
    rag_topic = INTENT_TO_TOPIC.get(intent, None)
    rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, rag_topic)
    print(f"--- 📚 RAG FETCHED: {len(rag_docs)} context snippets ---")
    
    # STEP 4: LLM Generation
    snapshot = get_snapshot(caller_id, intent)
    prompt = build_prompt(text_en, rag_docs, history, snapshot)
    
    response_en = await gpu_scheduler.run(engine.generate, prompt)
    print(f"--- 🤖 AI RESPONSE (EN): {response_en} ---")
    log_processing_step(call_id, "llm_generate", None, response_en)

    # STEP 5: Guardrails
    safety_response = apply_guardrails(response_en, intent, rag_docs, shared_detector)
    final_en = safety_response if safety_response else response_en
    if safety_response: print(f"--- 🛡️ GUARDRAIL TRIGGERED: Response modified ---")

    log_message(call_id, "ai", final_en)

    # STEP 6: Session Update (Safe Check)
    if session_store:
        new_history = history + [
            {"role": "user", "text": text_en},
            {"role": "ai", "text": final_en}
        ]
        session_store.update_session(phone, {"history": new_history[-6:]})
        session_store.persist_later(phone)

    # STEP 7: Translate Back to Malayalam
    reply_ml = await cpu_scheduler.run(en_to_ml, final_en)
    print(f"--- 🎙️ FINAL REPLY (ML): {reply_ml} ---\n")
    
    return reply_ml