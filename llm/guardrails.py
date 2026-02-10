# llm/guardrails.py
import re
from sentence_transformers import util

def apply_guardrails(response_en, intent, rag_docs, intent_detector):
    """
    Checks if the answer is factual and grounded in context.
    """
    # 1. Skip check for general greetings or if no docs found
    if intent == "general" or not rag_docs:
        return None

    # 2. Fact Check: Hallucinated Numbers
    numbers_in_response = re.findall(r'\d+', response_en)
    combined_context = " ".join(rag_docs)
    
    # Allow common years/numbers
    SAFE_NUMBERS = {"10", "12", "2024", "2025", "2026"}

    for num in numbers_in_response:
        if num in SAFE_NUMBERS:
            continue
        # Strict check: If the number isn't in the context, flag it.
        if num not in combined_context:
            return (
                "I don’t have verified numerical data for that at the moment. "
                "Please refer to the official admission notification."
            )

    # 3. Groundedness: Is the response actually related to the data we found?
    # Use the shared IntentDetector's model for efficiency
    try:
        res_emb = intent_detector.model.encode(response_en, convert_to_tensor=True)
        # Handle list of docs by encoding them and taking the average or max similarity
        ctx_emb = intent_detector.model.encode(rag_docs, convert_to_tensor=True)
        
        # Calculate similarity (Bot Answer vs. All Docs)
        # We take the max similarity (at least one doc should support the answer)
        scores = util.cos_sim(res_emb, ctx_emb)
        max_context_sim = scores.max().item()
        
        print(f"🛡️ GUARDRAIL SCORE: {max_context_sim:.4f}")

        if max_context_sim < 0.25: # Threshold (0.25 is safer for loose conversation)
            return "The official data for this query is currently being updated. May I help you with course details or placements instead?"
            
    except Exception as e:
        print(f"⚠️ Guardrail Error: {e}")
        return None

    return None