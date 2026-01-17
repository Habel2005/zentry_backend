import numpy as np
import onnxruntime as ort
import sounddevice as sd
from transformers import AutoTokenizer
import logging

class TTSModule:
    def __init__(self, model_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

    def tell(self, text, play=True, sr=16000):
        # [FIX] Handle empty text gracefully
        if not text or not text.strip():
            return np.zeros(sr, dtype=np.float32)

        try:
            inputs = self.tokenizer(text, return_tensors="np")
            
            # [FIX] Check if tokenizer failed
            if "input_ids" not in inputs or inputs["input_ids"] is None:
                logging.error(f"TTS Tokenizer failed for text: {text}")
                return np.zeros(sr, dtype=np.float32)

            ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64)}
            audio = self.session.run(None, ort_inputs)[0].squeeze().astype(np.float32)
            
            audio /= max(1e-5, abs(audio).max())

            if play:
                try:
                    sd.play(audio, sr)
                    sd.wait()
                except Exception:
                    pass
            return audio

        except Exception as e:
            logging.error(f"TTS Error: {e}")
            return np.zeros(sr, dtype=np.float32)