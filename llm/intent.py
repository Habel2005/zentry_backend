import numpy as np
from sentence_transformers import SentenceTransformer, util

class IntentDetector:
    def __init__(self):
        # Extremely small and fast (80MB), perfect for 50+ concurrent lookups
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define "Anchor" phrases for each intent
        self.intent_anchors = {
            "seat": ["how many seats are available", "vacancy in btech", "is there any spot left"],
            "placement": ["what is the average salary", "placement records", "companies visiting"],
            "eligibility": ["what is the qualification", "minimum marks required", "am i eligible"],
            "general": ["hello", "who are you", "tell me about the college"]
        }
        
        # Pre-compute embeddings for speed
        self.anchor_embeddings = {
            intent: self.model.encode(phrases, convert_to_tensor=True)
            for intent, phrases in self.intent_anchors.items()
        }

    def detect(self, text_en: str) -> str:
        query_embedding = self.model.encode(text_en, convert_to_tensor=True)
        best_intent = "general"
        max_sim = 0.42

        for intent, anchors in self.anchor_embeddings.items():
            # Check similarity against all phrases in this intent group
            similarities = util.cos_sim(query_embedding, anchors)
            top_sim = np.max(similarities.cpu().numpy())
            
            if top_sim > max_sim:
                max_sim = top_sim
                best_intent = intent
                
        return best_intent

# Singleton instance
detector = IntentDetector()
def detect_intent(text): return detector.detect(text)