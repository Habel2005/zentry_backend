
from llama_cpp import Llama

class PhiEngine:
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )

    def generate(self, prompt):
        result = self.llm(
            prompt,
            max_tokens=120,
            temperature=0.2,
            top_p=0.9,
            stop=["</s>"]
        )
        return result["choices"][0]["text"].strip()
