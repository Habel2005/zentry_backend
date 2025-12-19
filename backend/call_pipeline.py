import asyncio
import base64
import json
import logging
from backend.vad_stream import VADStreamer

# Assume 'stt' and 'tts' are passed in from main_server.py globally
# from backend.stt_worker import MalayalamSTT
# from tts.tts_module import TTSModule

class CallPipeline:
    def __init__(self, uuid, websocket, stt_engine, tts_engine):
        self.uuid = uuid
        self.ws = websocket
        self.stt = stt_engine
        self.tts = tts_engine
        self.vad = VADStreamer(min_energy=400) # Tuned for phone mic
        self.current_task = None
        self.is_responding = False

    async def handle_audio(self, chunk):
        result = self.vad.process_chunk(chunk)

        # 🛑 CASE 1: USER INTERRUPTED (Barge-In)
        if result == "BARGE_IN":
            if self.is_responding and self.current_task:
                print(f"[{self.uuid}] 🛑 BARGE-IN DETECTED! Stopping AI...")
                self.current_task.cancel() # Kill the LLM/TTS task immediately
                # Optional: Send "stop" JSON to FreeSWITCH to clear its buffer
                # await self.ws.send(json.dumps({"type": "stop"}))
            return

        # ✅ CASE 2: SENTENCE COMPLETED
        if isinstance(result, bytes):
            # Start a background task for this turn
            self.current_task = asyncio.create_task(self.run_conversation_turn(result))

    async def run_conversation_turn(self, audio_bytes):
        self.is_responding = True
        try:
            # 1. STT (Speech to Text)
            text = await self.stt.transcribe(audio_bytes)
            if not text or len(text) < 2: return
            print(f"[{self.uuid}] 🗣️ User: {text}")

            # 2. LLM (Get text response) -> REPLACE WITH YOUR LLM FUNC
            # response_text = await llm.generate(text) 
            response_text = f"നിങ്ങൾ പറഞ്ഞത് {text} എന്നല്ലേ?" # Echo back for testing

            # 3. TTS (Text to Speech)
            # Returns raw WAV bytes (16k, mono, 16-bit PCM)
            audio_data = self.tts.tell(response_text) 

            # 4. SEND TO FREESWITCH (Must be JSON Wrapped!)
            # FreeSWITCH mod_audio_stream expects base64 inside JSON
            b64_audio = base64.b64encode(audio_data).decode('utf-8')
            
            payload = {
                "type": "streamAudio", 
                "data": {
                    "audioDataType": "raw",
                    "sampleRate": 16000,
                    "audioData": b64_audio
                }
            }
            
            await self.ws.send(json.dumps(payload))
            print(f"[{self.uuid}] 🤖 AI Responded")

        except asyncio.CancelledError:
            print(f"[{self.uuid}] 🔇 Task Cancelled (User Interrupted)")
        except Exception as e:
            logging.error(f"Pipeline Error: {e}")
        finally:
            self.is_responding = False
            self.current_task = None

    async def cleanup(self):
        if self.current_task:
            self.current_task.cancel()