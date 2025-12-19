
from llm.engine import PhiEngine
from llm.scheduler import LLMScheduler
from llm.prompt_builder import build_prompt
from llm.rag import RAGRetriever
from session.session_store import SessionStore
from data.college_db import CollegeDB
from translate.translator import Translator

engine = PhiEngine("phi-4-mini-instruct.Q4_K_M.gguf")
scheduler = LLMScheduler()
rag = RAGRetriever()
sessions = SessionStore()
college = CollegeDB()
translator = Translator()

async def handle_llm(phone, ml_text):
    en = translator.translate(ml_text, "ml-en")
    session = sessions.load_or_create(phone)

    seat = college.get_seat_info("B.Tech") if "seat" in en.lower() else None
    placement = college.get_placement_info() if "placement" in en.lower() else None

    docs = rag.retrieve(en)
    prompt = build_prompt(en, docs, session["summary"], seat, placement)

    en_reply = await scheduler.run(engine, prompt)
    ml_reply = translator.translate(en_reply, "en-ml")

    sessions.update_summary(session["session_id"], session["summary"] + f"\n{en}")
    return ml_reply
