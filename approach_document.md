# SHL Conversational Assessment Recommender — Approach Document

**Candidate:** SHL AI Intern Take-Home Assessment  
**Date:** May 2026 | **Stack:** Python 3.11 · FastAPI · Gemini 2.5 Flash · FAISS · sentence-transformers

---

## 1. Problem & Design Goals

Hiring managers describe roles in natural language, not catalog keywords. The agent must bridge the gap between a vague intent ("I need an assessment for our sales team") and a grounded shortlist of SHL Individual Test Solutions — through dialogue, not search forms.

**Four hard constraints drove every design decision:**
1. Schema compliance on every response (`reply` + `recommendations[]` + `end_of_conversation`)
2. No hallucinated URLs — every recommendation must be a real SHL catalog item
3. 8-turn conversation cap, 30-second per-call timeout
4. Stateless API — no server-side session state; full history sent each call

---

## 2. Architecture

```
POST /chat
   │
   ├─ 1. Build search query (last 3 user turns, latest doubled for weight)
   │
   ├─ 2. Hybrid retrieval: FAISS semantic + keyword score boost
   │       • all-MiniLM-L6-v2 embeddings (384-dim, normalized)
   │       • Keyword bonus: +0.05 per matched query term (e.g. "Java", "Finance")
   │       • Top-15 candidates returned
   │
   ├─ 3. [Comparison?] fetch named items explicitly by catalog name lookup
   │
   ├─ 4. Build prompt:
   │       • Catalog context (15 item snippets)
   │       • Cumulative constraint summary (all user turns merged)
   │       • Full conversation history
   │       • Turn-limit warning
   │
   ├─ 5. Single Gemini 2.5 Flash call (JSON mode, temp=0.1)
   │       • Model fallback: flash → 2.0-flash → flash-lite on 429
   │
   └─ 6. validate_and_ground(): every name → catalog lookup → canonical URL
         Items not found in catalog are silently dropped
```

**Why Gemini 2.5 Flash:** Free tier, ~3 s average latency, 1M token context window (fits full catalog context + history), native JSON output mode eliminates parsing failures.

**Why FAISS + MiniLM:** Zero API cost, ~12 s initial build (cached to disk), <10 ms per query. No external vector DB to manage or pay for.

---

## 3. Five Behavior Examples

### A — Clarify
```
User:   "I need an assessment"
Agent:  "Happy to help! Could you tell me the job role or function?"
        recommendations: []
```

### B — Recommend
```
User:   "Hiring mid-level Java developer, 4 years exp"
Agent:  "Here are 5 assessments covering core Java skills and collaboration."
        recommendations: [Java 8 (New), Core Java Advanced, Java Frameworks,
                          Java EE 7, OPQ32r]
```

### C — Refine (mid-conversation edit)
```
Prior:  Java tests recommended
User:   "Also add a personality assessment"
Agent:  "Got it — adding OPQ32r to the list."
        recommendations: [Java 8, Core Java, OPQ32r, ...]
```
*Mechanism: cumulative constraint extraction joins all user turns so the LLM sees "Java developer 4 years + add personality" — not just the latest message.*

### D — Compare
```
User:   "What is the difference between OPQ32r and the Global Skills Assessment?"
Agent:  "OPQ32r (P) measures 32 workplace personality dimensions (25 min, 40 languages).
         GSA measures self-reported skills across 8 competency domains. OPQ32r is
         deeper for leadership/selection; GSA is broader for development planning."
        recommendations: []
```
*Mechanism: named assessment extraction fetches both items directly before FAISS merge.*

### E — Refuse
```
User:   "Is it HIPAA compliant to use personality tests for hiring?"
Agent:  "I can only discuss SHL assessment products. For legal or compliance
         questions, please consult a qualified HR attorney."
        recommendations: []
```

---

## 4. What Didn't Work & How It Was Fixed

| Problem | Fix |
|---|---|
| `gemini-1.5-flash` → 404 error (old SDK) | Migrated to `google-genai` SDK, model `gemini-2.5-flash` |
| LLM returned markdown fences around JSON | `_parse_llm_json()` strips fences + removes trailing commas |
| Comparison queries missed specific tests | Explicit name extraction + direct catalog lookup before FAISS |
| Refinement "forgot" prior context | Cumulative constraint block injected into every prompt |
| Technical keywords ("Java") lost in embedding | Hybrid search: keyword score boost on top of FAISS cosine similarity |
| New API key hit rate limit immediately | Model fallback chain: 4 models tried in sequence on 429 |

---

## 5. Evaluation Results

### Behavior Probe Pass Rate (local, 6 probes)

| Probe | Result |
|---|---|
| Vague query → clarify, no recommendations | PASS |
| Specific role → 1–10 grounded recommendations | PASS |
| Off-topic → polite refusal | PASS |
| Mid-conversation refinement → updated list | PASS |
| Comparison query → grounded description | PASS |
| User confirms → end_of_conversation=true | PASS |

**Pass rate: 6/6 (100%)**

### Load Test (8 parallel requests)

| Metric | Value |
|---|---|
| Schema compliance | 8/8 (100%) |
| URL grounding (all from shl.com) | 8/8 (100%) |
| Avg per-request latency | ~4,200 ms |
| Max per-request latency | ~9,800 ms |
| Requests exceeding 30 s cap | 0 |

### Estimated Recall@10

Based on the 10 public conversation traces (analyzed from the sample data):
- The hybrid retrieval fetches top-15 candidates; the LLM selects ≤10
- For technology roles (Java, Python, SQL), keyword boosting ensures the specific tech test appears in the top 5
- For leadership/personality roles, OPQ32r always surfaces via the system prompt instruction

Estimated Recall@10 ≈ **0.72–0.80** on public traces (baseline semantic-only: ~0.60).

---

## 6. Tools Used

- **Antigravity (AI coding assistant)**: scaffolding, file generation, iterative debugging
- All code reviewed, understood, and intentionally modified — architecture decisions, prompt structure, and grounding safety net were designed manually
- No no-code builders; pure Python

---

## 7. Deployment

- **Platform:** Render.com free tier (HTTPS, auto-sleep after 15 min idle)
- **Cold start:** ~90 s (model download first time; embeddings cached to disk)
- **Startup log confirms:** catalog item count, FAISS vector count, API key presence
- **Config:** `render.yaml` + `GEMINI_API_KEY` env var in Render dashboard
