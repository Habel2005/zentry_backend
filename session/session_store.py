# session/session_store.py
import asyncio
from supabase import create_client

class SessionStore:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
        self.cache = {}  # phone -> session dict
        self.queue = asyncio.Queue()

        # background sync task
        asyncio.create_task(self._sync_worker())

    def get_session(self, phone):
        if phone not in self.cache:
            # create in-memory session ONLY
            self.cache[phone] = {
                "summary": "",
                "last_intent": None
            }
        return self.cache[phone]

    def update_session(self, phone, data):
        self.cache[phone].update(data)

    def persist_later(self, phone):
        self.queue.put_nowait(phone)

    async def _sync_worker(self):
        while True:
            phone = await self.queue.get()
            session = self.cache.get(phone)
            if not session:
                continue

            # run blocking Supabase call off-loop
            await asyncio.to_thread(
                self.supabase.table("sessions")
                .upsert({"phone": phone, **session})
                .execute
            )
