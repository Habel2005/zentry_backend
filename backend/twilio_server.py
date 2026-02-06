import json
import base64
import audioop
from fastapi import FastAPI, WebSocket, Request
from .call_pipeline import CallPipeline
from backend.call_context import CallContext
from db.call_repo import start_call

app = FastAPI()

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    pipeline = None
    stream_sid = None

    # Retrieve models from app state
    # These will be set in main_server.py
    stt_instance = getattr(websocket.app.state, "stt", None)
    tts_instance = getattr(websocket.app.state, "tts", None)

    if not stt_instance or not tts_instance:
        print("❌ Error: STT/TTS models not initialized in app.state")
        await websocket.close()
        return

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)

            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                caller = data['start']['customParameters'].get('caller', 'unknown')

                print(f"📞 Incoming Twilio Call: {stream_sid} from {caller}")

                # Using your existing CallContext
                ctx = CallContext(uuid=stream_sid, phone=caller)

                # Initialize DB entry
                ctx.call_id, ctx.caller_id = start_call(stream_sid, ctx.phone)

                # Pass the FastAPI websocket to your pipeline
                pipeline = CallPipeline(ctx, websocket, stt_instance, tts_instance)
                pipeline.is_twilio = True # Mark as Twilio for format handling
                print(f"✅ Twilio Stream Attached: {stream_sid}")

            elif data['event'] == 'media' and pipeline:
                # 1. Decode Base64
                payload = data['media']['payload']
                chunk_mulaw = base64.b64decode(payload)

                # 2. Convert Mu-law (8kHz) -> PCM Linear (16-bit, 8kHz)
                # Twilio audio is always 8000Hz.
                chunk_pcm16 = audioop.ulaw2lin(chunk_mulaw, 2)

                # 3. Process through your existing VAD
                await pipeline.handle_audio(chunk_pcm16)

            elif data['event'] == 'stop':
                print(f"🛑 Twilio Stream Stopped: {stream_sid}")
                if pipeline: await pipeline.cleanup()
                break

    except Exception as e:
        print(f"⚠️ Twilio WebSocket Error: {e}")
    finally:
        if pipeline:
            await pipeline.cleanup()
