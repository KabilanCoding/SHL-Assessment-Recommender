# SHL Conversational Assessment Recommender

A FastAPI-based conversational agent that recommends SHL Individual Test assessments through a natural language dialogue.

## Quick Start (Local)

### 1. Install dependencies
```bash
cd "Conversational SHL"
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com/app/apikey
```

### 3. Run the server
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test it
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}]}'
```

### 5. Run local tests
```bash
python test_local.py
```

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "Sure. What is seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Got it. Here are 5 assessments that fit a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Architecture

```
POST /chat
   │
   ▼
Build search query (last 3 user turns)
   │
   ▼
FAISS semantic search → top-15 catalog items
   │  (sentence-transformers/all-MiniLM-L6-v2, no API cost)
   ▼
[Comparison query?] → also fetch named items explicitly
   │
   ▼
Single Gemini 1.5 Flash call
   (system prompt + catalog context + conversation history)
   │
   ▼
Parse JSON response
   │
   ▼
validate_and_ground()  ← CRITICAL SAFETY NET
   Every URL replaced with catalog URL; hallucinated names dropped
   │
   ▼
ChatResponse
```

## Test Type Codes

| Code | Category |
|------|----------|
| K | Knowledge & Skills |
| P | Personality & Behavior |
| S | Simulations |
| A | Ability & Aptitude |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |

## Deployment (Render.com)

1. Push this repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com), connect your repo.
3. Render auto-detects `render.yaml`.
4. In the Render dashboard → Environment → add `GEMINI_API_KEY`.
5. Deploy — your public URL will be `https://shl-recommender.onrender.com`.
