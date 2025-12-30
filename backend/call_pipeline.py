import asyncio
import base64
import json
import logging
from backend.vad_stream import VADStreamer
from llm.brain import handle_llm
from db.call_repo import log_message,end_call

class CallPipeline:
    def __init__(self, ctx, websocket, stt, tts):
        self.ws = websocket
        self.phone = self.ctx.phone
        self.uuid = self.ctx.uuid
        self.ctx = ctx
        self.stt = stt
        self.tts = tts
        self.vad = VADStreamer(min_energy=400)
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

    async def run_ai_turn(self, audio_bytes):
        self.is_responding = True
        try:
            # 1. STT (Wait for shared GPU slot)
            text_ml = await self.stt.transcribe(audio_bytes)
            if not text_ml or len(text_ml) < 2: return

            log_message(
                call_id=self.ctx.call_id,
                speaker="user",
                raw_text=text_ml
            )

            
            # 2. THE BRAIN (Delegated to your LLM module)
            # This handles: Translate -> Session -> RAG -> Phi-4 -> Translate Back
            reply_ml = await handle_llm(
                self.ctx.call_id,
                self.ctx.caller_id,
                self.ctx.phone,
                text_ml
            )


            print(f"[{self.uuid}] 🤖 {reply_ml}")

            # 3. TTS
            audio_data_np = await asyncio.to_thread(self.tts.tell, reply_ml, play=False)

            # CONVERT NUMPY FLOAT32 -> PCM INT16 BYTES
            audio_bytes = (audio_data_np * 32767).astype(np.int16).tobytes()

            # 4. SEND
            payload = {
                "type": "streamAudio",
                "data": {
                    "audioDataType": "raw", 
                    "sampleRate": 16000,
                    "audioData": base64.b64encode(audio_bytes).decode('utf-8') # Now this works
                }
            }
            await self.ws.send(json.dumps(payload))

        except asyncio.CancelledError:
            pass # Task was killed by a barge-in
        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
        finally:
            self.is_responding = False

    async def cleanup(self):
        if self.current_task: self.current_task.cancel()
        end_call(self.ctx.call_id)
