"""
Enhanced core agent logic for the SHL Assessment Recommender.

Improvements over v1:
  - Refinement memory: extract cumulative constraints from full conversation
    so mid-conversation edits merge with prior context, not restart.
  - Hybrid search: catalog.search_hybrid() combines semantic + keyword scoring.
  - Comparison mode: reliably extracts named assessments from the query and
    fetches them directly by name before merging with FAISS results.
  - Robust JSON parsing: handles markdown code fences, extra keys, trailing commas.
  - Model fallback chain: tries gemini-2.5-flash → gemini-2.0-flash → lite on 429.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types as genai_types

from app.catalog import catalog, item_to_snippet
from app.models import ChatRequest, ChatResponse, Message, Recommendation
from app.prompts import SYSTEM_PROMPT, build_prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client (new google-genai SDK)
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not _API_KEY:
    logger.warning("GEMINI_API_KEY not set — LLM calls will fail!")

_CLIENT = genai.Client(api_key=_API_KEY)

# Model priority — try each on quota/rate errors
_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]

_GENERATION_CONFIG = genai_types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    response_mime_type="application/json",
    temperature=0.1,        # very low: deterministic catalog grounding
    max_output_tokens=4096, # increased: prevent JSON truncation on large lists
)

_EXECUTOR = ThreadPoolExecutor(max_workers=8)

# ---------------------------------------------------------------------------
# Intent detection helpers
# ---------------------------------------------------------------------------
_COMPARISON_RE = re.compile(
    r"\b(differ|difference|compare|vs\.?|versus|between|which is better"
    r"|what.s the difference|how does .+ compare)\b",
    re.IGNORECASE,
)

_REFINE_RE = re.compile(
    r"\b(also add|add|include|remove|drop|exclude|instead|without|actually"
    r"|change|update|modify|plus|and also|on top of)\b",
    re.IGNORECASE,
)

_END_RE = re.compile(
    r"^(perfect|great|thanks|thank you|that.?s all|looks good|done"
    r"|that.?s what we need|yes|okay|ok|that works|sounds good)[.!?\s]*$",
    re.IGNORECASE,
)


def _is_comparison(text: str) -> bool:
    return bool(_COMPARISON_RE.search(text))


def _is_refinement(messages: list[Message]) -> bool:
    """True if there are prior recommendations and the user is modifying constraints."""
    if len(messages) < 3:
        return False
    # Check if any prior assistant message had recommendations
    for msg in messages[:-1]:
        if msg.role == "assistant":
            try:
                data = json.loads(msg.content)
                if data.get("recommendations"):
                    return True
            except Exception:
                # Content may be plain text in real usage — treat as refinement if
                # we have a multi-turn conversation
                if len(messages) >= 3:
                    return True
    return False


def _extract_named_assessments(text: str) -> list[str]:
    """
    Extract likely assessment names from a comparison query.
    Handles: 'OPQ32r vs GSA', 'compare OPQ and DSI', '"Java 8 (New)" versus ...'
    """
    # Quoted names first
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted

    # Split on comparison delimiters
    delimiters = r"\bvs\.?\b|\bversus\b|\bbetween\b|\bcompare\b|\band\b|\bto\b"
    parts = re.split(delimiters, text, flags=re.IGNORECASE)
    candidates = []
    for p in parts:
        p = p.strip().strip(".,?!")
        # Filter out very short fragments and stop words
        if len(p) > 3 and not re.match(
            r"^(the|what|is|difference|how|does|or|of)$", p, re.IGNORECASE
        ):
            candidates.append(p)
    return candidates


def _build_search_query(messages: list[Message]) -> str:
    """
    Build a search query from the last 3 user turns (weighted toward most recent).
    """
    user_texts = [m.content for m in messages if m.role == "user"]
    recent = user_texts[-3:]
    if recent:
        # Duplicate the last message to boost its weight in the FAISS query
        recent = recent + [recent[-1]]
    return " ".join(recent)


def _extract_cumulative_context(messages: list[Message]) -> str:
    """
    Build a structured requirements summary from the FULL conversation.
    This is the refinement-memory mechanism: when user says 'also add X',
    the LLM sees all prior constraints and can merge them.
    """
    if len(messages) <= 1:
        return ""

    user_turns = [m.content for m in messages if m.role == "user"]
    if len(user_turns) <= 1:
        return ""

    # Build a numbered list of what the user has said
    lines = []
    for i, text in enumerate(user_turns, 1):
        lines.append(f"Turn {i}: {text}")

    return (
        "The user has provided the following information across the conversation:\n"
        + "\n".join(lines)
        + "\n\nCombine ALL these details when recommending or refining."
    )


# ---------------------------------------------------------------------------
# Catalog grounding safety net
# ---------------------------------------------------------------------------
def _validate_and_ground(raw_recs: list[dict]) -> list[Recommendation]:
    """
    For every LLM-produced recommendation:
      1. Look up the name in the catalog (exact → fuzzy → URL fallback).
      2. Replace URL with the canonical catalog URL.
      3. Derive test_type from catalog keys[].
      4. Silently drop anything not found in the catalog.

    This is the hard guarantee: every URL returned is a real SHL catalog URL.
    """
    valid: list[Recommendation] = []
    seen: set[str] = set()

    for rec in raw_recs[:10]:
        name = (rec.get("name") or "").strip()
        url = (rec.get("url") or "").strip()
        if not name:
            continue

        item = catalog.resolve_recommendation(name, url)
        if item is None:
            logger.debug("Dropped non-catalog recommendation: %r", name)
            continue

        canonical = item["name"]
        if canonical.lower() in seen:
            continue
        seen.add(canonical.lower())

        valid.append(
            Recommendation(
                name=canonical,
                url=item["link"],
                test_type=catalog.primary_code(item),
            )
        )

    return valid


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------
def _parse_llm_json(raw: str) -> dict:
    """
    Parse LLM output to a dict. Handles:
    - Clean JSON
    - JSON wrapped in ```json ... ``` fences
    - Trailing commas (common LLM mistake)
    - Truncated JSON (max_tokens hit mid-array): recovers partial recommendations
    """
    raw = raw.strip()

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas: ,] or ,}
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Truncation recovery: find and close the outermost {...} block
    # by counting brackets and closing any that are still open
    block_match = re.search(r"\{", raw)
    if block_match:
        candidate = raw[block_match.start():]
        # Trim to last complete object in the recommendations array
        # by finding the last complete '}' before truncation
        last_complete = candidate.rfind("}")
        if last_complete > 0:
            candidate = candidate[: last_complete + 1]
            # Now count open/close braces and brackets to see what's needed
            open_b = candidate.count("{")
            close_b = candidate.count("}")
            open_sq = candidate.count("[")
            close_sq = candidate.count("]")
            # Close any open arrays first, then objects
            suffix = "]" * max(0, open_sq - close_sq) + "}" * max(0, open_b - close_b)
            try:
                candidate_fixed = re.sub(r",\s*$", "", candidate) + suffix
                return json.loads(candidate_fixed)
            except json.JSONDecodeError:
                pass

    logger.error("Unparseable LLM response (first 600 chars): %s", raw[:600])
    raise ValueError("LLM response is not valid JSON")


# ---------------------------------------------------------------------------
# LLM call with model fallback chain
# ---------------------------------------------------------------------------
def _call_llm(prompt: str) -> dict:
    """
    Call Gemini with automatic fallback across _MODELS on 429 quota errors.
    Non-quota errors are raised immediately.
    """
    last_exc: Exception | None = None

    for model_name in _MODELS:
        try:
            logger.info("LLM call → model: %s", model_name)
            response = _CLIENT.models.generate_content(
                model=model_name,
                contents=prompt,
                config=_GENERATION_CONFIG,
            )
            return _parse_llm_json(response.text)

        except Exception as exc:
            err = str(exc)
            if any(
                kw in err
                for kw in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE",
                           "quota", "rate limit", "high demand", "overloaded")
            ):
                logger.warning("Transient error on %s — trying next model", model_name)
                last_exc = exc
                time.sleep(2)
                continue
            raise  # non-quota error — propagate immediately

    logger.error("All models exhausted. Last error: %s", last_exc)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------
async def run_agent(request: ChatRequest) -> ChatResponse:
    """Stateless agent — called once per POST /chat."""
    import asyncio

    messages = request.messages
    turn_count = sum(1 for m in messages if m.role == "assistant")
    last_user = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    # --- Hard turn cap ---
    force_end = turn_count >= 7

    # --- Build search query (recent user turns, latest doubled for weight) ---
    query = _build_search_query(messages)

    # --- Hybrid retrieval ---
    retrieved = catalog.search_hybrid(query, top_k=15)

    # --- Comparison: also fetch named items explicitly ---
    if _is_comparison(last_user):
        named = _extract_named_assessments(last_user)
        logger.info("Comparison query detected — fetching: %s", named)
        named_items = catalog.get_by_names(named)
        # Prepend named items (they must appear in context for comparison)
        seen_links = {it["link"] for it in named_items}
        for it in retrieved:
            if it["link"] not in seen_links:
                named_items.append(it)
                seen_links.add(it["link"])
        retrieved = named_items[:15]

    snippets = [item_to_snippet(it) for it in retrieved]

    # --- Refinement memory: cumulative context from full conversation ---
    cumulative_ctx = _extract_cumulative_context(messages)

    # --- Build prompt ---
    prompt = build_prompt(messages, snippets, turn_count, cumulative_ctx)

    # --- LLM call (blocking → thread pool) ---
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_EXECUTOR, _call_llm, prompt)
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
        return ChatResponse(
            reply=(
                "I'm experiencing a temporary issue. Please try again in a moment."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # --- Extract + validate ---
    reply: str = data.get("reply") or "Could you rephrase that? I want to make sure I help you correctly."
    raw_recs: list[dict] = data.get("recommendations") or []
    end_of_conv: bool = bool(data.get("end_of_conversation", False))

    recommendations = _validate_and_ground(raw_recs)

    if force_end:
        end_of_conv = True

    logger.info(
        "Response: intent_recs=%d grounded=%d eoc=%s",
        len(raw_recs), len(recommendations), end_of_conv,
    )

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conv,
    )
