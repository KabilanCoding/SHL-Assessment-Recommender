"""
Startup entry point — loads .env then launches uvicorn.
Run with:  python start.py
"""
import os
from pathlib import Path

# Load .env before anything else
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"[start] Loaded .env from {env_file}")
except ImportError:
    pass

# Validate API key
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key or api_key == "your_gemini_api_key_here":
    print("\n❌  ERROR: GEMINI_API_KEY is not set.")
    print("   1. Get a free key at: https://aistudio.google.com/app/apikey")
    print("   2. Edit .env and set: GEMINI_API_KEY=your_key_here")
    print("   3. Run this script again.\n")
    raise SystemExit(1)

print(f"[start] GEMINI_API_KEY set (prefix: {api_key[:8]}…)")

import uvicorn
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=False,
    workers=1,
    log_level="info",
)
