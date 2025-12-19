import asyncio
import logging
import signal
from backend.audio_server import start_audio_server
from backend.esl_client import run_esl_client
from backend.stt_worker import MalayalamSTT
from tts.tts_module import TTSModule

# Global Shared Resources (Load Once)
logging.basicConfig(level=logging.INFO)

async def shutdown(loop, signal=None):
    if signal:
        logging.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [t.cancel() for t in tasks]
    logging.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    logging.error(f"Caught exception: {msg}")

if __name__ == "__main__":
    # 1. Initialize Shared AI Models (Pass these to your servers)
    print("⏳ Loading AI Models (this may take 30s)...")
    stt = MalayalamSTT("models/ct2-whisper-medium")
    tts = TTSModule("models/mms-tts-mal.onnx")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 2. Define the tasks
    # Task A: WebSocket Server for Audio (Listens on 5001)
    audio_task = start_audio_server(stt, tts, port=5001)
    
    # Task B: ESL Client for Control (Connects to FS:8021)
    esl_task = run_esl_client(host="127.0.0.1", port=8021, password="ClueCon")

    # 3. Signals for graceful exit
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(loop, s)))
    loop.set_exception_handler(handle_exception)

    print("🚀 Zentry AI System Started. Waiting for calls...")
    try:
        loop.run_until_complete(asyncio.gather(audio_task, esl_task))
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
        print("🛑 System Shutdown Complete.")