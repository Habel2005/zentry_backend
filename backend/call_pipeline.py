import asyncio
import base64
import json
import logging
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message,end_call

# backend/call_pipeline.py

class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ctx = ctx # [FIX] Move this to the top
        self.ws = websocket
        self.phone = self.ctx.phone
        self.uuid = self.ctx.uuid
        self.stt = stt
        self.tts = tts

        # Use the probability-based VAD (v5) without 'min_energy'
        self.vad = VADStreamer(sample_rate=8000)
        self.current_task = None
        self.is_responding = False

    async def handle_audio(self, chunk):
        result = self.vad.process_chunk(chunk)

        if result == "BARGE_IN":
            if self.is_responding and self.current_task:
                print(f"[{self.uuid}] 🛑 Barge-in: Cancelling AI response")
                self.current_task.cancel()
            return

        if isinstance(result, bytes):
            # Run the AI turn in a task we can cancel if interrupted
            self.current_task = asyncio.create_task(self.run_ai_turn(result))

# In backend/call_pipeline.py

    async def run_ai_turn(self, audio_bytes):
        self.is_responding = True
        try:
            # 1. Transcribe
            text_ml = await self.stt.transcribe(audio_bytes, sample_rate=8000)
            
            # Use strip() to ignore silent/empty Whisper output
            if not text_ml or len(text_ml.strip()) < 2:
                return

            # Log user message to DB
            log_message(call_id=self.ctx.call_id, speaker="user", raw_text=text_ml)
            
            # 2. Call the new Brain with debug prints
            reply_ml = await handle_llm(
                self.ctx.call_id,
                self.ctx.caller_id,
                self.ctx.phone,
                text_ml
            )

            if not reply_ml or not reply_ml.strip():
                return

            # 3. Generate and Stream TTS
            audio_data_np = await asyncio.to_thread(self.tts.tell, reply_ml, play=False)
            
            if audio_data_np is not None:
                # Convert to 16-bit PCM and send via WebSocket
                audio_bytes_out = (audio_data_np * 32767).astype(np.int16).tobytes()
                payload = {
                    "type": "streamAudio",
                    "data": {
                        "audioDataType": "raw", 
                        "sampleRate": 16000,
                        "audioData": base64.b64encode(audio_bytes_out).decode('utf-8')
                    }
                }
                await self.ws.send(json.dumps(payload))

        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
        finally:
            self.is_responding = False

    async def cleanup(self):
        if self.current_task: self.current_task.cancel()
        end_call(self.ctx.call_id)
