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

    def tell(self, text, play=True, sr=8000): # Default to 8000 for Phone
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
            
            # [FIX] Downsample to 8000Hz if requested (Simple Decimation)
            if sr == 8000:
                audio = audio[::2] 

            # Normalize
            max_val = abs(audio).max()
            if max_val > 0:
                audio = audio / max(1e-5, max_val)
            
            # Debug Stats
            print(f"🔊 TTS Gen ({sr}Hz): {len(audio)} samples, Max Amp: {max_val:.4f}")

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