# llm/intent.py
def detect_intent(text_en: str) -> str:
    t = text_en.lower()
    if "seat" in t or "available" in t:
        return "seat"
    if "placement" in t or "salary" in t:
        return "placement"
    if "eligible" in t or "qualification" in t:
        return "eligibility"
    return "general"
