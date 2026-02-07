import logging

# Global variable to store the instance
_translator_instance = None

def get_translator():
    global _translator_instance
    
    if _translator_instance is None:
        print("🔄 Initializing IndicTrans2 (Lazy Load)...")
        try:
            # Import inside here to prevent circular imports
            from translate.translator import Translator
            _translator_instance = Translator()
            print("✅ IndicTrans2 Ready!")
        except Exception as e:
            logging.error(f"❌ Failed to load Translator: {e}")
            raise e
            
    return _translator_instance

def ml_to_en(text):
    if not text: return ""
    try:
        # FIX: Call .translate() with the direction string
        return get_translator().translate(text, direction="ml-en")
    except Exception as e:
        print(f"⚠️ Translation Error (ML->EN): {e}")
        return text 

def en_to_ml(text):
    if not text: return ""
    try:
        # FIX: Call .translate() with the direction string
        return get_translator().translate(text, direction="en-ml")
    except Exception as e:
        print(f"⚠️ Translation Error (EN->ML): {e}")
        return text