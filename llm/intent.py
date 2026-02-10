# llm/intent.py
from sentence_transformers import SentenceTransformer, util

class IntentDetector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.intent_anchors = {
            "seat": [
                "how many seats", "vacancy", "is computer science full", 
                "admission availability", "seat matrix", "management quota slot"
            ],
            "fee": [
                "fee structure", "how much money", "cost of engineering", 
                "semester payment", "scholarship", "financial"
            ],
            "placement": [
                "placement packages", "average salary", "companies visiting",
                "google amazon recruitment", "highest package", "jobs"
            ],
            "eligibility": [
                "cut off marks", "minimum percentage", "plus two marks",
                "entrance exam rank", "can i join with low marks", "requirements"
            ],
            # [NEW] Explicit Intent for Courses/Branches
            "courses": [
                "which branches", "available courses", "computer science", 
                "mechanical engineering", "what programs", "departments",
                "list of groups", "streams"
            ],
            "general": [
                "hello", "where is the college", "hostel facility", 
                "transportation", "location", "contact number"
            ]
        }
        
        # Pre-compute embeddings
        self.anchor_embeddings = {
            k: self.model.encode(v) for k, v in self.intent_anchors.items()
        }

    def detect(self, text):
        if not text or len(text.strip()) < 2:
            return "general"
            
        query_emb = self.model.encode(text)
        best_intent = "general"
        max_score = -1
        
        for intent, anchors in self.anchor_embeddings.items():
            scores = util.cos_sim(query_emb, anchors)[0]
            score = float(scores.max())
            
            if score > max_score:
                max_score = score
                best_intent = intent
        
        return best_intent if max_score > 0.35 else "general"

# Singleton
detector = IntentDetector()

def detect_intent(text):
    return detector.detect(text)