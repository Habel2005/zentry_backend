
SYSTEM_PROMPT = """
You are an official college admission assistant.
Rules:
- Never invent numbers, dates, or guarantees.
- If data is unavailable, say it is not officially published.
- Keep responses short and factual.
"""

def build_prompt(user_en, rag_docs, session_summary, seat_info, placement_info):
    ctx = []
    if rag_docs:
        ctx.append("Admission Info:\n" + "\n".join(rag_docs))
    if seat_info:
        ctx.append(f"Seat Info: Remaining seats {seat_info['remaining']}")
    if placement_info:
        ctx.append("Placement Info: " + placement_info["message"])

    return f"""{SYSTEM_PROMPT}

Conversation summary:
{session_summary}

Context:
{chr(10).join(ctx)}

User:
{user_en}

Answer:
"""
