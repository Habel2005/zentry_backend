import asyncio
from faster_whisper import WhisperModel

class MalayalamSTT:
    def __init__(self, model_path):
        print(f"⚙️ Loading Whisper Model: {model_path}")
        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16" # Use int8_float16 if VRAM is tight
        )
        # Allow max 3 concurrent GPU inferences. 
        # The 4th call will wait asynchronously (non-blocking) until one finishes.
        self.gpu_lock = asyncio.Semaphore(3)

    async def transcribe(self, audio_bytes):
        # We use asyncio.to_thread to run the blocking model.transcribe 
        # in a separate thread, but governed by the asyncio Semaphore.
        async with self.gpu_lock:
            return await asyncio.to_thread(self._sync_transcribe, audio_bytes)

    def _sync_transcribe(self, audio_bytes):
        # Whisper expects float32 array, faster_whisper handles bytes often, 
        # but robust way is numpy.
        import numpy as np
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0
        
        segments, _ = self.model.transcribe(audio_array, language="ml", beam_size=1)
        return " ".join(s.text for s in segments).strip()