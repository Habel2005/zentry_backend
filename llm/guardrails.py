from sentence_transformers import util
import re

def apply_guardrails(response_en, intent, rag_docs, intent_detector):
    """
    Checks if the answer is factual and grounded in context.
    """
    # 1. Skip check for general greetings
    if intent == "general":
        return None

    # 2. Fact Check: Hallucinated Numbers
    # If the response contains a percentage or large number not in RAG, flag it.
    numbers_in_response = re.findall(r'\d+', response_en)
    combined_context = " ".join(rag_docs)
    
    SAFE_NUMBERS = {"10", "12", "2024", "2025"}

    for num in numbers_in_response:
        if num in SAFE_NUMBERS:
            continue
        if num not in combined_context:
            return (
                "I don’t have verified numerical data for that at the moment. "
                "Please refer to the official admission notification."
            )


    # 3. Groundedness: Is the response actually related to the data we found?
    # Use the shared IntentDetector's model for efficiency
    res_emb = intent_detector.model.encode(response_en, convert_to_tensor=True)
    ctx_emb = intent_detector.model.encode(rag_docs, convert_to_tensor=True)
    
    # If the bot's answer is totally unrelated to the provided documents (similarity < 0.5)
    max_context_sim = util.cos_sim(res_emb, ctx_emb).max().item()
    
    if max_context_sim < 0.5:
        # Fallback response instead of a made-up one
        return "The official data for this query is currently being updated. May I help you with course details or placements instead?"

    return None