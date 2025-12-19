# llm/scheduler.py
import asyncio

class LLMScheduler:
    def __init__(self):
        self.sem = asyncio.Semaphore(2)

    async def run(self, fn, *args):
        async with self.sem:
            return await asyncio.to_thread(fn, *args)
