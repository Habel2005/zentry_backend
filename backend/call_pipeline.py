import asyncio
import base64
import json
import logging
import os
import traceback
import numpy as np
import time
import audioop
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message, end_call

# Load assets into RAM once (Global Cache)
ASSETS = {}
def load_assets():
    if ASSETS: return
    asset_dir = "assets"
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
        print("⚠️ 'assets/' folder missing. Please create it and add .wav files.")
        return

    for filename in os.listdir(asset_dir):
        if filename.endswith(".wav"):
            # Load as raw bytes (assuming 16kHz PCM 16-bit Mono)
            with open(os.path.join(asset_dir, filename), "rb") as f:
                # Skip 44-byte WAV header to get raw PCM
                data = f.read()[44:] 
                ASSETS[filename.split(".")[0]] = data
    print(f"✅ Loaded {len(ASSETS)} Reflex Audio Assets")

load_assets()

class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ctx = ctx
        self.ws = websocket
        self.stt = stt
        self.tts = tts
        self.is_twilio = False
        
        # [TUNED] Lower threshold to 0.4 for Twilio Phone Audio
        self.vad = VADStreamer(sample_rate=16000, threshold=0.2, min_energy=0.015)
        self.processing_lock = asyncio.Lock()

    async def handle_audio(self, chunk):
        # 1. STRICT LOCK: Walkie-Talkie mode
        if self.processing_lock.locked():
            return

        result = self.vad.process_chunk(chunk)
        if result == "BARGE_IN":
            return 

        if isinstance(result, bytes):
            if len(result) > 4000: 
                asyncio.create_task(self.execute_turn(result))

    async def send_to_twilio(self, pcm_audio_bytes):
        """
        Convert TTS Audio (Usually 22k/16k) -> 8k Mu-Law for Twilio
        """
        # Assume TTS output is 16000Hz (Based on TTSModule config)
        in_rate = 16000
        out_rate = 8000

        # 1. Downsample (using audioop for speed)
        # ratecv returns (new_bytes, state). We ignore state for single chunks or keep None
        pcm_8k, _ = audioop.ratecv(pcm_audio_bytes, 2, 1, in_rate, out_rate, None)

        # 2. Convert Linear PCM -> Mu-law
        mulaw_data = audioop.lin2ulaw(pcm_8k, 2)

        # 3. Base64 Encode
        payload = base64.b64encode(mulaw_data).decode('utf-8')

        # 4. JSON Packet
        message = {
            "event": "media",
            "streamSid": self.ctx.uuid,
            "media": {"payload": payload}
        }

        await self.ws.send_text(json.dumps(message))

    async def execute_turn(self, audio_bytes):
        if self.processing_lock.locked(): return
        
        async with self.processing_lock:
            try:
                # 1. STT
                text_ml = await self.stt.transcribe(audio_bytes, sample_rate=16000)
                if not text_ml or len(text_ml.strip()) < 2: return

                # Log User Input
                log_message(call_id=self.ctx.call_id, speaker="user", raw_text=text_ml)
                
                # 2. BRAIN (Returns Tuple)
                # response_type: "text" | "reflex"
                # content: Malayalam Text OR Asset Filename (e.g., "intro")
                # log_text: English/Malayalam text for debug logs
                response_type, content, log_text = await handle_llm(
                    self.ctx.call_id,
                    self.ctx.caller_id,
                    self.ctx.phone,
                    text_ml
                )
                
                if not content: return

                # 3. EXECUTION
                if response_type == "reflex":
                    print(f"⚡ REFLEX ACTIVATE: Playing {content}.wav")
                    await self.play_asset(content) # Defined in previous turn
                
                else:
                    print(f"⏳ Generating TTS for: {log_text[:20]}...")
                    # TTS
                    audio_data_np = await asyncio.to_thread(self.tts.tell, content, play=False, sr=16000)
                    
                    if audio_data_np is None: return

                    # Stream
                    audio_bytes_total = (audio_data_np * 32767).astype(np.int16).tobytes()
                    CHUNK_SIZE = 1280
                    for i in range(0, len(audio_bytes_total), CHUNK_SIZE):
                        chunk = audio_bytes_total[i:i+CHUNK_SIZE]
                        await self.send_to_twilio(chunk)
                        await asyncio.sleep(0.04)

            except Exception as e:
                logging.error(f"Pipeline Error: {e}")

    async def cleanup(self):
        print(f"🧹 Cleaning up call {self.ctx.call_id}...")
        try:
            end_call(self.ctx.call_id)
        except Exception as e:
            print(f"Cleanup error: {e}")
