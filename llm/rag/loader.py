import json
from pathlib import Path
from PyPDF2 import PdfReader

def load_txt(path):
    return Path(path).read_text(encoding="utf-8")

def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def load_json(path):
    data = json.loads(Path(path).read_text())
    return json.dumps(data, indent=2)

def load_file(path):
    if path.endswith(".pdf"):
        return load_pdf(path)
    if path.endswith(".json"):
        return load_json(path)
    return load_txt(path)
