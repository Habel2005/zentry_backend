from supabase import create_client
import uuid
from config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class SessionStore:
    def load_or_create(self, phone):
        res = supabase.table("sessions").select("*").eq("phone", phone).limit(1).execute()
        if res.data:
            return res.data[0]
        sid = str(uuid.uuid4())
        supabase.table("sessions").insert({"session_id": sid, "phone": phone, "summary": ""}).execute()
        return {"session_id": sid, "summary": ""}

    def update_summary(self, sid, summary):
        supabase.table("sessions").update({"summary": summary}).eq("session_id", sid).execute()
