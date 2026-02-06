import uvicorn
import logging
from backend.stt_worker import MalayalamSTT
from tts.tts_module import TTSModule
from backend.twilio_server import app

# Global Shared Resources (Load Once)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # 1. Initialize Shared AI Models
    print("⏳ Loading AI Models (this may take 30s)...")
    stt = MalayalamSTT("models/whisper")
    tts = TTSModule("models/tts/tts_mal.onnx")
    
    # 2. Attach models to FastAPI app state
    app.state.stt = stt
    app.state.tts = tts

    print("🧠 Initializing Memory...")
    # sessions = SessionStore(url="SUPABASE_URL", key="SUPABASE_KEY")
    # brain.init_globals(sessions)
    
    print("🚀 Zentry AI System Starting (Twilio Mode)...")
    
    # 3. Start Uvicorn
    # Using 'app' object directly, not string, so state is preserved
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")
