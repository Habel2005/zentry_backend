import asyncio
import base64
import json
import logging
import traceback
import numpy as np
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message, end_call

class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ctx = ctx
        self.ws = websocket
        self.phone = self.ctx.phone
        self.uuid = self.ctx.uuid
        self.stt = stt
        self.tts = tts
        self.vad = VADStreamer()
        self.current_task = None
        self.is_responding = False

    async def handle_audio(self, chunk):
        result = self.vad.process_chunk(chunk)
        if result == "BARGE_IN":
            if self.is_responding and self.current_task:
                print(f"\n[{self.uuid}] 🛑 Barge-in: Cancelling AI response")
                self.current_task.cancel()
            return
        if isinstance(result, bytes):
            if len(result) > 4000:
                self.current_task = asyncio.create_task(self.run_ai_turn(result))

    async def run_ai_turn(self, audio_bytes):
        self.is_responding = True
        try:
            print(f"\n⚡ Processing {len(audio_bytes)} bytes of audio...")
            text_ml = await self.stt.transcribe(audio_bytes, sample_rate=8000)
            if not text_ml or len(text_ml.strip()) < 2: return

            log_message(call_id=self.ctx.call_id, speaker="user", raw_text=text_ml)
            
            reply_ml = await handle_llm(
                self.ctx.call_id,
                self.ctx.caller_id,
                self.ctx.phone,
                text_ml
            )
            if not reply_ml: return

            audio_data_np = await asyncio.to_thread(self.tts.tell, reply_ml, play=False, sr=8000)
            if audio_data_np is None or len(audio_data_np) == 0: return

            audio_bytes_total = (audio_data_np * 32767).astype(np.int16).tobytes()
            
            # [CRITICAL FIX] Standard VoIP Packet Size
            # 8000Hz * 2 bytes = 16000 bytes/sec
            # 20ms = 320 bytes (The standard chunk size for phone calls)
            CHUNK_SIZE = 320 
            
            # Sleep slightly LESS than 0.02s to keep the buffer full and prevent stutter
            # 0.018s sleep ensures we send data slightly faster than real-time
            SLEEP_TIME = 0.018 
            
            print(f"📤 Streaming {len(audio_bytes_total)} bytes (Raw PCM 8k) to client...")
            
            for i in range(0, len(audio_bytes_total), CHUNK_SIZE):
                chunk = audio_bytes_total[i:i+CHUNK_SIZE]
                try:
                    await self.ws.send(chunk)
                    await asyncio.sleep(SLEEP_TIME) 
                except Exception:
                    break

            print("✅ Audio Stream Complete")

        except asyncio.CancelledError:
            print("🚫 Task Cancelled")
        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
            traceback.print_exc()
        finally:
            self.is_responding = False

    async def cleanup(self):
        if self.current_task: self.current_task.cancel()
        end_call(self.ctx.call_id)