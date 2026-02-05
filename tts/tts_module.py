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
        print(f"✅ TTS Module Loaded: {model_path}")

    def tell(self, text, play=True, sr=8000): 
        if not text or not text.strip():
            return np.zeros(sr, dtype=np.float32)

        try:
            inputs = self.tokenizer(text, return_tensors="np")
            if "input_ids" not in inputs:
                return np.zeros(sr, dtype=np.float32)

            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64)
            }
            
            # Model generates 16000Hz natively
            audio = self.session.run(None, ort_inputs)[0].squeeze().astype(np.float32)

            # [FIX] Smart Normalization
            # If audio is silent (all zeros), max_val is 0.
            max_val = abs(audio).max()
            
            if max_val > 0.01: # Only normalize if there is actual audio
                audio = audio / max_val

            mean_val = abs(audio).mean()
            if mean_val < 0.01:
                print(f"⚠️ TTS WARNING: Audio is very quiet! (Mean: {mean_val:.4f})")
            
            # Debug Stats
            print(f"🔊 TTS Gen ({sr}Hz): {len(audio)} samples, Peak Amp: {max_val:.4f}")

            return audio

        except Exception as e:
            logging.error(f"TTS Error: {e}")
            return np.zeros(sr, dtype=np.float32)