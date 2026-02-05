import asyncio
import logging

class ESLClient:
    def __init__(self, host, port, password):
        self.host = host
        self.port = port
        self.password = password
        self.reader = None
        self.writer = None

    async def connect(self):
        while True:
            try:
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
                logging.info("✅ ESL Connected to FreeSWITCH")
                
                # Authenticate
                await self.send_cmd(f"auth {self.password}")
                await self.read_response()
                
                # Subscribe to events
                await self.send_cmd("event plain CHANNEL_ANSWER CHANNEL_HANGUP_COMPLETE")
                
                # Event Loop
                await self.listen()
            except Exception as e:
                logging.error(f"⚠️ ESL Connection Failed: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def send_cmd(self, cmd):
        self.writer.write((cmd + "\n\n").encode())
        await self.writer.drain()

    async def read_response(self):
        data = await self.reader.readuntil(b"\n\n")
        return data.decode().strip()

    async def listen(self):
        while True:
            # 1. Read the "Envelope" Headers
            try:
                line = await self.reader.readuntil(b"\n\n")
            except asyncio.IncompleteReadError:
                break
                
            headers = self.parse_headers(line.decode())
            
            # 2. Read the "Letter" Body
            event_data = {}
            if "Content-Length" in headers:
                length = int(headers["Content-Length"])
                body = await self.reader.readexactly(length)
                # [FIX 1] Parse the body to get the ACTUAL event details
                event_data = self.parse_headers(body.decode())

            # 3. Extract Info (Look in body first, then headers)
            event_name = event_data.get("Event-Name", headers.get("Event-Name"))
            uuid = event_data.get("Unique-ID", headers.get("Unique-ID"))
            
            if event_name == "CHANNEL_ANSWER":
                            # Get the UUID and Phone Number from headers or body
                            uuid = event_data.get("Unique-ID", headers.get("Unique-ID"))
                            phone = event_data.get("Caller-Caller-ID-Number", headers.get("Caller-Caller-ID-Number", "unknown"))
                            
                            logging.info(f"📞 Call Answered: {uuid} from {phone} -> Starting Audio Stream")
                            
                            # [FIX] Send a valid JSON string. 
                            # Note the double braces {{ }} to escape them in the f-string 
                            # and the inner double quotes \" for JSON compatibility.
                            json_metadata = f'{{"uuid": "{uuid}", "caller": "{phone}"}}'
                            
                            cmd = f"api uuid_audio_stream {uuid} start ws://127.0.0.1:5001 mono 8000 {json_metadata}"
                            await self.send_cmd(cmd)

            elif event_name == "CHANNEL_HANGUP_COMPLETE":
                logging.info(f"❌ Call Ended: {uuid}")

    def parse_headers(self, data):
        headers = {}
        for line in data.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                headers[k] = v.strip() # Added strip() to clean up values
        return headers

async def run_esl_client(host, port, password):
    client = ESLClient(host, port, password)
    await client.connect()