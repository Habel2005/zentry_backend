# session/session_store.py
import asyncio

class SessionStore:
    def __init__(self, url=None, key=None):
        # self.supabase = create_client(url, key)
        self.cache = {} 
        self.queue = asyncio.Queue()
        asyncio.create_task(self._sync_worker())

    def get_session(self, phone):
        if phone not in self.cache:
            self.cache[phone] = {
                "history": [],  # List of {"role": "...", "text": "..."}
                "metadata": {}
            }
        return self.cache[phone]

    def update_session(self, phone, data):
        if phone in self.cache:
            self.cache[phone].update(data)

    def persist_later(self, phone):
        self.queue.put_nowait(phone)

    # session/session_store.py (Update _sync_worker)
    async def _sync_worker(self):
        while True:
            phone = await self.queue.get()
            session = self.cache.get(phone)
            
            # Only sync if supabase client exists
            if not session or not hasattr(self, 'supabase') or not self.supabase:
                continue
                
            await asyncio.to_thread(
                self.supabase.table("sessions").upsert({"phone": phone, **session}).execute
            )

    async def flush_all(self):
        for phone in list(self.cache.keys()):
            await self._persist(phone)

