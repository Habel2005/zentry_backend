# llm/brain.py
from llm.engine import PhiEngine
from llm.scheduler import LLMScheduler
from llm.intent import detect_intent
from llm.guardrails import apply_guardrails
from llm.prompt import build_prompt
from llm.cache import get as cache_get, set as cache_set
from llm.rag.retriever import retrieve
from llm.translate import ml_to_en, en_to_ml

engine = PhiEngine("models/phi-4-mini-instruct.Q4_K_M.gguf")
scheduler = LLMScheduler()

async def handle_llm(phone: str, text_ml: str) -> str:
    # 1. Translate in
    text_en = ml_to_en(text_ml)

    # 2. Intent
    intent = detect_intent(text_en)

    # 3. Guardrails
    guard = apply_guardrails(intent)
    if guard:
        return en_to_ml(guard)

    # 4. Cache
    key = f"{intent}:{text_en.lower()}"
    cached = cache_get(key)
    if cached:
        return en_to_ml(cached)

    # 5. RAG
    context = retrieve(text_en)

    # 6. Prompt
    prompt = build_prompt(text_en, context)

    # 7. LLM
    response_en = await scheduler.run(engine.generate, prompt)

    # 8. Cache
    cache_set(key, response_en)

    # 9. Translate out
    return en_to_ml(response_en)
