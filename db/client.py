# db/client.py
import os
from supabase import create_client
from dotenv import load_dotenv

# Load variables from the .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# [FIX] Check for invalid SSL_CERT_FILE environment variable
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if not os.path.exists(cert_path):
        # If the file doesn't exist, remove the variable so Python
        # falls back to the default system certificate store.
        del os.environ["SSL_CERT_FILE"]

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None

def init_supabase():
    global supabase
    if not supabase:
        if not SUPABASE_URL or not SUPABASE_KEY:
             raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_KEY in .env file")
        
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase