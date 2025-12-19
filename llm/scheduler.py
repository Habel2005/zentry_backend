
import asyncio

class LLMScheduler:
    def __init__(self, max_concurrent=2):
        self.sem = asyncio.Semaphore(max_concurrent)

    async def run(self, engine, prompt):
        async with self.sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, engine.generate, prompt)
