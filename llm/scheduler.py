import asyncio

class AsyncScheduler:
    def __init__(self, max_concurrent=1):
        self.sem = asyncio.Semaphore(max_concurrent)

    async def run(self, fn, *args):
        async with self.sem:
            # This moves the blocking CPU work to a separate thread
            # keeping your Audio Loop free!
            return await asyncio.to_thread(fn, *args)

# Create two separate instances
# GPU Scheduler: Strict limit (e.g., 1 or 2) to prevent OOM
gpu_scheduler = AsyncScheduler(max_concurrent=1) 

# CPU Scheduler: Higher limit (e.g., 4 or 8) for translations
# Your i9 can easily handle 4 concurrent translations.
cpu_scheduler = AsyncScheduler(max_concurrent=4)