import asyncio
import base64
import json
import logging
import traceback
import numpy as np
import time
import audioop
import os
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message, end_call

# --- 1. GLOBAL ASSET LOADER ---
# Loads WAV files into RAM once so we don't read disk on every call.
ASSETS = {}

def load_assets():
    if ASSETS: return
    asset_dir = "assets"
    if not os.path.exists(asset_dir):
        print("⚠️ 'assets/' folder missing. Reflex audio will fail.")
        return

    print("⏳ Loading Audio Assets...")
    for filename in os.listdir(asset_dir):
        if filename.endswith(".wav"):
            path = os.path.join(asset_dir, filename)
            with open(path, "rb") as f:
                # [CRITICAL] Skip 44-byte WAV header to get raw PCM data
                # If we don't do this, you'll hear a "pop" or static at the start.
                data = f.read()[44:] 
                
                # Store key as "intro", "fallback" (remove .wav)
                key = filename.split(".")[0]
                ASSETS[key] = data
                print(f"   🔹 Loaded asset: '{key}' ({len(data)} bytes)")
    print(f"✅ Loaded {len(ASSETS)} Reflex Assets")

# Run loader immediately when this file is imported
load_assets()


class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ctx = ctx
        self.ws = websocket
        self.stt = stt
        self.tts = tts
        self.is_twilio = False
        
        # [TUNED] VAD Settings for Phone Audio
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
        Convert 16kHz PCM -> 8kHz Mu-Law for Twilio
        """
        try:
            # 1. Downsample 16000Hz -> 8000Hz
            pcm_8k, _ = audioop.ratecv(pcm_audio_bytes, 2, 1, 16000, 8000, None)

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
        except Exception as e:
            print(f"⚠️ Streaming Error: {e}")

    # --- 2. THE MISSING FUNCTION ---
    async def play_asset(self, asset_name):
        """
        Fast Path: Plays a pre-loaded raw PCM buffer.
        """
        if asset_name not in ASSETS:
            print(f"❌ Asset '{asset_name}' not found in {list(ASSETS.keys())}")
            return

        print(f"⚡ REFLEX ACTIVATE: Streaming '{asset_name}'...")
        raw_audio = ASSETS[asset_name]
        
        # Stream in 40ms chunks (same as TTS)
        # 16000Hz * 0.04s * 2 bytes = 1280 bytes
        CHUNK_SIZE = 1280
        
        try:
            for i in range(0, len(raw_audio), CHUNK_SIZE):
                chunk = raw_audio[i:i+CHUNK_SIZE]
                await self.send_to_twilio(chunk)
                await asyncio.sleep(0.04) # Real-time pacing
        except Exception as e:
            print(f"⚠️ Asset Playback Error: {e}")


    async def execute_turn(self, audio_bytes):
        if self.processing_lock.locked(): return
        
        async with self.processing_lock:
            try:
                print(f"\n🔒 Pipeline Locked. Processing {len(audio_bytes)} bytes...")
                
                # 1. STT
                text_ml = await self.stt.transcribe(audio_bytes, sample_rate=16000)
                if not text_ml or len(text_ml.strip()) < 2: return

                log_message(call_id=self.ctx.call_id, speaker="user", raw_text=text_ml)
                
                # --- ⚡ SMART FILLER LOGIC ---
                # If text is long (> 15 chars) and NOT a greeting, play "wait.wav"
                # This fills the silence while the GPU works.
                is_greeting = len(text_ml) < 15 or text_ml.strip() in ["ഹലോ", "ഹായ്", "hello"]
                
                if not is_greeting:
                    print("⏳ Long query detected. Playing filler audio...")
                    # We use create_task so it plays in background/parallel if possible,
                    # but since we are locked, we just await it to ensure user hears it first.
                    await self.play_asset("wait")

                # 2. BRAIN
                response_type, content, log_text = await handle_llm(
                    self.ctx.call_id,
                    self.ctx.caller_id,
                    self.ctx.phone,
                    text_ml
                )
                
                if not content: return

                # 3. EXECUTION
                if response_type == "reflex":
                    # If we already played 'wait', playing 'intro' might sound weird,
                    # but 'reflex' usually happens on short texts where we skipped 'wait'.
                    await self.play_asset(content)
                
                else:
                    # TTS
                    print(f"⏳ Generating TTS for: {log_text[:20]}...")
                    audio_data_np = await asyncio.to_thread(self.tts.tell, content, play=False, sr=16000)
                    
                    if audio_data_np is None: return

                    audio_bytes_total = (audio_data_np * 32767).astype(np.int16).tobytes()
                    CHUNK_SIZE = 1280
                    for i in range(0, len(audio_bytes_total), CHUNK_SIZE):
                        chunk = audio_bytes_total[i:i+CHUNK_SIZE]
                        await self.send_to_twilio(chunk)
                        await asyncio.sleep(0.04)

                print("✅ Turn Complete.")

            except Exception as e:
                logging.error(f"Pipeline Error: {e}")
                traceback.print_exc()

    async def cleanup(self):
        print(f"🧹 Cleaning up call {self.ctx.call_id}...")
        try:
            end_call(self.ctx.call_id)
        except Exception as e:
            print(f"Cleanup error: {e}")