SYSTEM_PROMPT = """
You are an official admission assistant for Zentry College.

Rules:
- Do not invent numbers, dates, or guarantees.
- If information is not in the context, say it is not officially available.

Context:
{context}

History:
{history}

User: {user_input}
Assistant:
"""


def build_prompt(user_en, rag_docs, history_list):
    # 1. RAG Context
    context_str = "No specific documents found."
    if rag_docs:
        context_str = "\n".join(rag_docs)

    # 2. Format History (Last 3 turns to save context window)
    # history_list = [{"role": "user", "text": "..."}, {"role": "ai", "text": "..."}]
    history_str = ""
    for turn in history_list[-6:]: # Keep last 6 exchanges (3 turns)
        role = "User" if turn["role"] == "user" else "Assistant"
        history_str += f"{role}: {turn['text']}\n"

    # 3. Fill Template
    return SYSTEM_PROMPT.format(
        context=context_str,
        history=history_str,
        user_input=user_en
    )