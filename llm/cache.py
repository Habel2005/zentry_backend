# llm/cache.py
from functools import lru_cache

@lru_cache(maxsize=100)
def response_cache(key):
    return None
