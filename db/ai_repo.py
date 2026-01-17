# db/ai_repo.py
from db.client import init_supabase

def log_processing_step(call_id, step_type, input_data=None, output_data=None, status="success", latency_ms=None):
    sb = init_supabase()
    sb.table("ai_processing_steps").insert({
        "call_id": call_id,
        "step_type": step_type,
        "input": input_data,
        "output": output_data,
        "status": status,
        "latency_ms": latency_ms
    }).execute()


def log_intent(call_id, intent, confidence=None):
    sb = init_supabase()
    sb.table("call_intents").insert({
        "call_id": call_id,
        "intent": intent,
        "confidence": confidence
    }).execute()


