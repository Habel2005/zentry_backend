# llm/brain.py
from .intent import detect_intent, detector as shared_detector
from llm.engine import PhiEngine
from llm.scheduler import LLMScheduler
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.rag.rag import RAGRetriever
from llm.translate import ml_to_en, en_to_ml
from session.session_store import SessionStore

# Initialize Singletons
engine = PhiEngine("models/phi-4-mini-instruct.Q4_K_M.gguf")
scheduler = LLMScheduler(max_concurrent=2)
rag = RAGRetriever()
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

    text_en = ml_to_en(text_ml)
    intent = detect_intent(text_en)
    rag_docs = rag.retrieve(text_en)

    prompt = build_prompt(text_en, rag_docs, history)

    response_en = await scheduler.run(engine.generate, prompt)

    safety_response = apply_guardrails(
        response_en, intent, rag_docs, shared_detector
    )
    final_en = safety_response if safety_response else response_en

    new_history = history + [
        {"role": "user", "text": text_en},
        {"role": "ai", "text": final_en}
    ]

    session_store.update_session(phone, {"history": new_history[-6:]})
    session_store.persist_later(phone)

    return en_to_ml(final_en)
