# db/snapshot_repo.py
from db.client import init_supabase

def get_snapshot(caller_id: str, intent: str):
    """
    Returns a SMALL operational snapshot string
    to guide the LLM (never raw numbers).
    """
    sb = init_supabase()

    notes = []

    # 1. Repeat caller signal
    caller = sb.table("caller_profiles") \
        .select("total_calls") \
        .eq("id", caller_id) \
        .execute()

    if caller.data and caller.data[0]["total_calls"] > 1:
        notes.append("Repeat caller. Previous enquiry exists.")


    # 3. Admission baseline (VERY CAREFUL)
    baseline = sb.table("admission_baseline") \
        .select("estimated_range, confidence_level") \
        .eq("quota_type", _map_intent_to_quota(intent)) \
        .order("date", desc=True) \
        .limit(1) \
        .execute()

    if baseline.data:
        confidence = baseline.data[0]["confidence_level"]
        if confidence in ("low", "medium"):
            notes.append("Admission availability is limited. Avoid guarantees.")

    if not notes:
        return "No special operational constraints."

    return " ".join(notes)


def _map_intent_to_quota(intent: str):
    if intent in ("management", "seat"):
        return "management"
    if intent == "nri":
        return "nri"
    return "general"
