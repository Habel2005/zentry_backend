# llm/translate.py
from translate.translator import Translator

translator = Translator()

def ml_to_en(text: str) -> str:
    return translator.translate(text, "ml-en")

def en_to_ml(text: str) -> str:
    return translator.translate(text, "en-ml")
