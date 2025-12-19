# llm/engine.py
from llama_cpp import Llama

class PhiEngine:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=2,         
            n_gpu_layers=40,      # feed GPU aggressively
            verbose=False
        )

    def generate(self, prompt):
        out = self.llm(
            prompt,
            max_tokens=120,
            temperature=0.4,      # slightly conversational
            top_p=0.9,
            repeat_penalty=1.1
        )
        return out["choices"][0]["text"].strip()
