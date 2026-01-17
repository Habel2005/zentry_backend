from llama_cpp import Llama

class PhiEngine:
    def __init__(self, model_path):
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,      # Increased to 4 for i9; 2 is a bit slow
            n_gpu_layers=40,   # Keep 40 since you have a 3080 Ti
            verbose=False
        )

    def generate(self, prompt: str) -> str:
        out = self.model(
            prompt,
            max_tokens=120,
            temperature=0.4,
            top_p=0.9,
            repeat_penalty=1.1,
            # [CRITICAL FIX] Stop the AI from hallucinating the whole script
            stop=["User:", "Assistant:", "\nUser", "<|end|>", "User :", "Assistant :"]
        )
        
        text = out["choices"][0]["text"].strip()

        # [SAFETY CLEANUP] Remove any leftover headers if the stop tokens missed them
        if "User:" in text:
            text = text.split("User:")[0].strip()
        if "Assistant:" in text:
            text = text.split("Assistant:")[0].strip()

        return text