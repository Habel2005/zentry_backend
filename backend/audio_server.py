import asyncio
import websockets
import json
from .call_pipeline import CallPipeline
from backend.call_context import CallContext
from db.call_repo import start_call

async def audio_handler(websocket, stt, tts):
    pipeline = None
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                uuid, phone = data.get("uuid"), data.get("caller", "unknown")
                if uuid:
                    
                    #doing the context
                    ctx = CallContext(uuid, phone)
                    ctx.call_id, ctx.caller_id = start_call(uuid, phone)

                    pipeline = CallPipeline(ctx, websocket, stt, tts)
                    print(f"✅ Stream Attached: {uuid}")
            elif isinstance(message, bytes) and pipeline:
                await pipeline.handle_audio(message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        if pipeline: await pipeline.cleanup()

async def start_audio_server(stt, tts):
    # Pass shared engines into the handler
    async with websockets.serve(lambda ws: audio_handler(ws, stt, tts), "0.0.0.0", 5001):
        await asyncio.Future() # Run forever