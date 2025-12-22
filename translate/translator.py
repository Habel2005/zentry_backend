import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
import re
import os

# 🔒 PRODUCTION TIP: We use CPU for translation to save 3080 Ti VRAM 
# for the STT (Whisper) and LLM (Phi-4). On an i9, this is sub-200ms.
DEVICE = "cpu"

# Domain-specific keyword mapping for Zentry Admission Assistant
PRE_MAP = {"ബി.ടെക്ക്": "B.Tech", "എം.സിഎ": "MCA", "എം.ടെക്": "M.Tech"}
POST_MAP = {v: k for k, v in PRE_MAP.items()}

class Translator:
    def __init__(self, directions=("ml-en", "en-ml")):
        """
        Loads distilled 200M models for high-concurrency translation.
        """
        self.models = {}
        self.tokenizers = {}
        self.directions = directions
        self.ip = IndicProcessor(inference=True)

        for direction in directions:
            if direction == "ml-en":
                model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
                src_lang, tgt_lang = "mal_Mlym", "eng_Latn"
            elif direction == "en-ml":
                model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
                src_lang, tgt_lang = "eng_Latn", "mal_Mlym"
            else:
                continue

            print(f"🔄 Loading {direction} translation model: {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True
            ).to(DEVICE)

            self.models[direction] = model
            self.tokenizers[direction] = tokenizer
            setattr(self, f"{direction}_src_lang", src_lang)
            setattr(self, f"{direction}_tgt_lang", tgt_lang)

        print("✅ Translator initialized successfully on CPU.\n")

    def _pre_map(self, text: str, direction: str) -> str:
        # Protect specific academic terms from being distorted by translation
        if direction == "ml-en":
            for k, v in PRE_MAP.items():
                text = re.sub(rf"{re.escape(k)}[\u0D00-\u0D7F]*", v, text)
        return text

    def _post_map(self, text: str, direction: str) -> str:
        # Restore Malayalam terms in the output
        if direction == "en-ml":
            for k, v in POST_MAP.items():
                text = re.sub(re.escape(k), v, text)
        return re.re.sub(r"<.*?>", "", text).strip()

    def translate(self, text: str, direction="ml-en") -> str:
        """
        Translates a single utterance. Used by handle_llm in the brain logic.
        """
        if direction not in self.models or not text.strip():
            return text

        model = self.models[direction]
        tokenizer = self.tokenizers[direction]
        src_lang = getattr(self, f"{direction}_src_lang")
        tgt_lang = getattr(self, f"{direction}_tgt_lang")

        # 1. Domain Mapping
        text_pre = self._pre_map(text, direction)
        
        # 2. IndicTrans Pre-processing
        batch = self.ip.preprocess_batch([text_pre], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        # 3. Inference
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128, # Increased for descriptive AI responses
                num_beams=1,        # Beam=1 for maximum speed in production
                do_sample=False
            )

        # 4. Post-processing
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        translated = self.ip.postprocess_batch(decoded, lang=tgt_lang)[0]
        return self._post_map(translated, direction)