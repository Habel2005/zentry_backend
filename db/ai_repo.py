# db/ai_repo.py
from db.client import init_supabase

# db/ai_repo.py
import json

def log_processing_step(call_id, step_type, input_data=None, output_data=None, status="success", latency_ms=None):
    sb = init_supabase()
    
    # Safety Wrapper: Ensure input is JSON-compatible
    if input_data and not isinstance(input_data, (dict, list)):
        input_data = {"raw_content": str(input_data)}
        
    if output_data and not isinstance(output_data, (dict, list)):
        output_data = {"raw_content": str(output_data)}

    sb.table("ai_processing_steps").insert({
        "call_id": call_id,
        "step_type": step_type,
        "input": input_data,  # Now guaranteed to be JSON object
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


