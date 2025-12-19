import asyncio
import websockets
import json
import logging
from backend.call_pipeline import CallPipeline

async def audio_handler(websocket, stt, tts):
    pipeline = None
    try:
        async for message in websocket:
            if isinstance(message, str):
                # Metadata / Handshake
                data = json.loads(message)
                # mod_audio_stream sometimes sends raw JSON first
                uuid = data.get("uuid") 
                if not uuid: continue # Skip non-uuid messages
                
                print(f"🔗 Audio Stream Linked: {uuid}")
                pipeline = CallPipeline(uuid, websocket, stt, tts)

            elif isinstance(message, bytes) and pipeline:
                # Raw Audio L16 16k
                await pipeline.handle_audio(message)
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logging.error(f"Audio Handler Error: {e}")
    finally:
        if pipeline:
            await pipeline.cleanup()

async def start_audio_server(stt, tts, port=5001):
    # Partial function to pass models into the handler
    handler = lambda ws: audio_handler(ws, stt, tts)
    async with websockets.serve(handler, "0.0.0.0", port):
        print(f"📡 Audio Server Listening on :{port}")
        await asyncio.Future() # Run forever