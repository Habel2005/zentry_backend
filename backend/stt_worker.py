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

    async def transcribe(self, audio_bytes, sample_rate=16000):
        # We use asyncio.to_thread to run the blocking model.transcribe 
        # in a separate thread, but governed by the asyncio Semaphore.
        async with self.gpu_lock:
            return await asyncio.to_thread(self._sync_transcribe, audio_bytes, sample_rate)

    def _sync_transcribe(self, audio_bytes, sample_rate):
        import numpy as np
        
        # 1. Convert bytes -> float32 array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).flatten().astype(np.float32) / 32768.0

        # 2. Resample if needed (Whisper expects 16k)
        if sample_rate != 16000:
            num_samples = len(audio_array)
            target_num_samples = int(num_samples * 16000 / sample_rate)
            # Use basic linear interpolation (fast, sufficient for STT)
            audio_array = np.interp(
                np.linspace(0.0, 1.0, target_num_samples, endpoint=False),
                np.linspace(0.0, 1.0, num_samples, endpoint=False),
                audio_array
            )

        # 3. Transcribe
        segments, _ = self.model.transcribe(audio_array, language="ml", beam_size=1)
        return " ".join(s.text for s in segments).strip()