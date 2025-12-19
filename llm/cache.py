# llm/cache.py
import time

_cache = {}

TTL = 180  # seconds

def get(key):
    val = _cache.get(key)
    if not val:
        return None
    text, ts = val
    if time.time() - ts > TTL:
        _cache.pop(key, None)
        return None
    return text

def set(key, value):
    _cache[key] = (value, time.time())
