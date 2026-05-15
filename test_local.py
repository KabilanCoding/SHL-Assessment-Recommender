"""
Quick local test script — runs without pytest.
Tests schema compliance, grounding, and basic behavior.

Usage:
    python test_local.py
"""
from __future__ import annotations
import asyncio, json, os, sys
from pathlib import Path
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

# Load env vars from .env if present
from dotenv import load_dotenv
load_dotenv()

from app.catalog import catalog
from app.models import ChatRequest, Message
from app.agent import run_agent

# ---------------------------------------------------------------------------
TESTS = [
    # (label, messages, assertions)
    (
        "Vague query — should clarify, no recommendations",
        [Message(role="user", content="I need an assessment")],
        {"recommendations_empty": True, "end_of_conversation_false": True},
    ),
    (
        "Complete query — Java developer, mid-level",
        [
            Message(role="user", content="Hiring a mid-level Java developer with 4 years experience"),
        ],
        {"has_recommendations": True, "max_10": True, "end_of_conversation_false": True},
    ),
    (
        "Senior leadership CXO — personality focus",
        [
            Message(role="user", content="We need a solution for senior leadership — CXOs, directors, 15+ years experience"),
            Message(role="assistant", content='{"reply": "Sure. Is this for selection or development?", "recommendations": [], "end_of_conversation": false}'),
            Message(role="user", content="Selection — comparing candidates against a leadership benchmark"),
        ],
        {"has_recommendations": True, "max_10": True},
    ),
    (
        "Off-topic refusal — legal question",
        [
            Message(role="user", content="Is it legal under HIPAA to use personality tests for hiring healthcare workers?"),
        ],
        {"recommendations_empty": True},
    ),
    (
        "Comparison query",
        [
            Message(role="user", content="What is the difference between OPQ32r and the Global Skills Assessment?"),
        ],
        {},  # just check it doesn't crash and returns valid schema
    ),
    (
        "Refine — add personality test",
        [
            Message(role="user", content="Hiring admin assistants who use Excel and Word"),
            Message(role="assistant", content='{"reply": "Here are some tests for admin assistants.", "recommendations": [{"name": "Microsoft Excel (New)", "url": "https://www.shl.com/products/product-catalog/view/microsoft-excel-new/", "test_type": "K"}], "end_of_conversation": false}'),
            Message(role="user", content="Actually, also add a personality assessment"),
        ],
        {"has_recommendations": True},
    ),
]


async def run_tests():
    print("Loading catalog …")
    catalog.load()
    print(f"Catalog ready: {len(catalog.items)} items\n")

    passed = 0
    failed = 0

    for label, messages, assertions in TESTS:
        print("-"*60)
        print(f"TEST: {label}")
        try:
            req = ChatRequest(messages=messages)
            resp = await run_agent(req)

            # Schema check
            assert isinstance(resp.reply, str) and resp.reply, "reply must be non-empty string"
            assert isinstance(resp.recommendations, list), "recommendations must be list"
            assert isinstance(resp.end_of_conversation, bool), "end_of_conversation must be bool"
            assert len(resp.recommendations) <= 10, "max 10 recommendations"

            # Grounding check — every URL must be from catalog
            catalog_links = {it["link"] for it in catalog.items}
            for rec in resp.recommendations:
                assert rec.url in catalog_links, f"HALLUCINATED URL: {rec.url}"
                assert rec.name, "name must not be empty"
                assert rec.test_type, "test_type must not be empty"

            # Specific assertions
            if assertions.get("recommendations_empty"):
                assert len(resp.recommendations) == 0, f"Expected empty recs, got {[r.name for r in resp.recommendations]}"
            if assertions.get("has_recommendations"):
                assert len(resp.recommendations) > 0, "Expected recommendations, got none"
            if assertions.get("end_of_conversation_false"):
                assert not resp.end_of_conversation, "Expected end_of_conversation=False"

            print("  PASS")

            print(f"  Reply: {resp.reply[:120]}...")
            if resp.recommendations:
                print(f"  Recommendations ({len(resp.recommendations)}):")
                for r in resp.recommendations:
                    print(f"    - {r.name} [{r.test_type}]")
            passed += 1

        except AssertionError as e:
            print(f"  FAIL: {e}")

            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")

            failed += 1

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed out of {len(TESTS)} tests")
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_tests())
    sys.exit(0 if ok else 1)
