# llm/guardrails.py
def apply_guardrails(intent: str):
    if intent == "placement":
        return (
            "Our college has a strong placement environment with reputed recruiters. "
            "Exact placement statistics are officially shared during counseling."
        )
    return None
