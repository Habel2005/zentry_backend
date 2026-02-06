import uvicorn
import logging

# Global Shared Resources (Load Once)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("🚀 Starting Zentry AI (Twilio Edition)...")
    # Just run the app. usage: file_name:app_instance_name
    uvicorn.run("backend.twilio_server:app", host="0.0.0.0", port=5001, log_level="info", reload=False)
