import asyncio
import base64
import json
import logging
import traceback
import numpy as np
import time
import audioop
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message, end_call

class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ctx = ctx
        self.ws = websocket
        self.stt = stt
        self.tts = tts
        self.is_twilio = False
        
        # [TUNED] 8kHz VAD settings
        # Note: Even though we upsample to 16k for STT, VAD might still be tuned for 8k or adaptable.
        # Since we are passing 16k audio to handle_audio now (from twilio_server), we should probably
        # ensure VAD expects 16k or we need to pass the original 8k chunk.
        # The user instructions said: "3. Send clean 16k audio to VAD/STT".
        # So we assume VADStreamer can handle 16k or we should update it.
        # Looking at previous code, VADStreamer was init with sample_rate=8000.
        # Let's update this to 16000 since we are upsampling.
        self.vad = VADStreamer(sample_rate=16000, threshold=0.6, min_energy=0.015)
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
                print(f"\n🔒 Pipeline Locked. Processing {len(audio_bytes)} bytes...")
                
                # --- STT ---
                # Now receiving 16k audio, so ensure STT knows it
                text_ml = await self.stt.transcribe(audio_bytes, sample_rate=16000)
                
                if not text_ml or len(text_ml.strip()) < 2: 
                    print("⚠️ Ignored empty STT.")
                    return

                log_message(call_id=self.ctx.call_id, speaker="user", raw_text=text_ml)
                
                # --- LLM ---
                reply_ml = await handle_llm(
                    self.ctx.call_id,
                    self.ctx.caller_id,
                    self.ctx.phone,
                    text_ml
                )
                if not reply_ml: return

                # --- TTS ---
                print("⏳ Generating TTS...")
                # TTS generates 16000Hz audio
                audio_data_np = await asyncio.to_thread(self.tts.tell, reply_ml, play=False, sr=16000)
                
                if audio_data_np is None or len(audio_data_np) == 0:
                    print("❌ TTS Error: Empty audio.")
                    return

                # --- CONVERSION ---
                audio_bytes_total = (audio_data_np * 32767).astype(np.int16).tobytes()
                
                # --- STREAMING ---
                print(f"📤 Streaming {len(audio_bytes_total)} bytes to phone...")

                if self.is_twilio:
                    # Source is 16kHz (32000 bytes/s). We want 40ms chunks.
                    # 0.04 * 32000 = 1280 bytes
                    CHUNK_SIZE = 1280
                    for i in range(0, len(audio_bytes_total), CHUNK_SIZE):
                        chunk = audio_bytes_total[i:i+CHUNK_SIZE]
                        await self.send_to_twilio(chunk)
                        await asyncio.sleep(0.04)
                else:
                    # Legacy/Debug socket path
                    CHUNK_SIZE = 320
                    SLEEP_TIME = 0.02

                    for i in range(0, len(audio_bytes_total), CHUNK_SIZE):
                        chunk = audio_bytes_total[i:i+CHUNK_SIZE]
                        try:
                            await self.ws.send(chunk)
                            await asyncio.sleep(SLEEP_TIME)
                        except Exception as e:
                            print(f"⚠️ Socket Dropped: {e}")
                            break
                
                print("✅ Turn Complete. Unlocking...")

            except Exception as e:
                logging.error(f"Pipeline Error: {e}")
                traceback.print_exc()

    async def cleanup(self):
        print(f"🧹 Cleaning up call {self.ctx.call_id}...")
        try:
            end_call(self.ctx.call_id)
        except Exception as e:
            print(f"Cleanup error: {e}")
