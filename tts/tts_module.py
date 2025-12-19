
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from transformers import AutoTokenizer

class TTSModule:
    def __init__(self, model_path, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mal")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

    def tell(self, text, play=True, sr=16000):
        inputs = self.tokenizer(text, return_tensors="np")
        ort_inputs = {"input_ids": inputs["input_ids"].astype(np.int64)}
        audio = self.session.run(None, ort_inputs)[0].squeeze().astype(np.float32)
        audio /= max(1e-5, abs(audio).max())

        if play:
            sd.play(audio, sr)
            sd.wait()
        return audio
