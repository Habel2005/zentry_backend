# backend/call_context.py
class CallContext:
    def __init__(self, uuid: str, phone: str):
        self.uuid = uuid
        self.phone = phone
        self.call_id = None
        self.caller_id = None
