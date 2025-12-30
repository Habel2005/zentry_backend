# llm/brain.py
import asyncio
from llm.intent import detect_intent, detector as shared_detector
from llm.engine import PhiEngine
from llm.scheduler import gpu_scheduler, cpu_scheduler
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.rag.retriever import RAGRetriever
from llm.rag.embedder import embedder_instance # Import the Global Singleton
from llm.translate import ml_to_en, en_to_ml
from session.session_store import SessionStore
from db.call_repo import log_message
from db.ai_repo import log_processing_step, log_intent
from db.snapshot_repo import get_snapshot

# 1. Initialize Singletons correctly
engine = PhiEngine("models/phi-4-mini-instruct.Q4_K_M.gguf")

# CRITICAL FIX: Pass the shared embedder to the retriever
rag = RAGRetriever(embedder_instance=embedder_instance)

session_store = None 

# 2. Topic Mapping (Bridges Intent -> RAG)
# Maps the 'intent' string to the 'topic' field in your ChromaDB metadata
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
    session = session_store.get_session(phone)
    history = session.get("history", [])[-6:]

    # ---------------------------------------------------------
    # STEP 1: Translate (CPU Bound)
    # ---------------------------------------------------------
    text_en = await cpu_scheduler.run(ml_to_en, text_ml)
    
    log_processing_step(call_id, "translate_ml_en", text_ml, text_en)

    # ---------------------------------------------------------
    # STEP 2: Intent Detection (Fast & First)
    # ---------------------------------------------------------
    # Must run BEFORE RAG to enable filtering
    intent = await cpu_scheduler.run(detect_intent, text_en)

    log_intent(call_id, intent)

    snapshot = get_snapshot(caller_id, intent)

    # ---------------------------------------------------------
    # STEP 3: Smart RAG Retrieval (Topic Filtered)
    # ---------------------------------------------------------
    # If intent is 'general', topic is None (searches all docs)
    rag_topic = INTENT_TO_TOPIC.get(intent, None)
    
    # Pass the topic to narrow down the search
    rag_docs = await cpu_scheduler.run(rag.retrieve, text_en, rag_topic)

    log_processing_step(call_id, "rag", text_en, [d[:80] for d in rag_docs])

    # ---------------------------------------------------------
    # STEP 4: Build Prompt
    # ---------------------------------------------------------
    prompt = build_prompt(text_en, rag_docs, history, snapshot)

    # ---------------------------------------------------------
    # STEP 5: LLM Generation (GPU Bound)
    # ---------------------------------------------------------
    response_en = await gpu_scheduler.run(engine.generate, prompt)

    log_processing_step(call_id, "llm_generate", None, response_en)

    # ---------------------------------------------------------
    # STEP 6: Guardrails & Translate Back
    # ---------------------------------------------------------
    # CRITICAL FIX: Pass 'shared_detector' as the 4th argument
    safety_response = apply_guardrails(response_en, intent, rag_docs, shared_detector)

    log_processing_step(
        call_id,
        "guardrail",
        status="modified" if safety_response else "passed"
    )

    final_en = safety_response if safety_response else response_en

    log_message(call_id, "ai", final_en)

    # Update History
    new_history = history + [
        {"role": "user", "text": text_en},
        {"role": "ai", "text": final_en}
    ]
    session_store.update_session(phone, {"history": new_history[-6:]})
    session_store.persist_later(phone)

    # Final Translation
    return await cpu_scheduler.run(en_to_ml, final_en)