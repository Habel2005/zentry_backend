# llm/scheduler.py
import asyncio

class LLMScheduler:
    def __init__(self, max_concurrent=2):
        self.sem = asyncio.Semaphore(max_concurrent)

    async def run(self, fn, *args):
        async with self.sem:
            return await asyncio.to_thread(fn, *args)
