"""
Staggered load test for the SHL Assessment Recommender.

Sends 8 POST /chat requests with 3-second gaps (realistic evaluator pattern).
The SHL evaluator runs one conversation at a time (sequential turns), so
staggered sequential testing is the correct benchmark.

Usage:
    python load_test.py [base_url]
    python load_test.py http://localhost:8000
    python load_test.py https://your-render-url.onrender.com
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

PAYLOADS = [
    {
        "label": "Vague query (clarify)",
        "body": {"messages": [{"role": "user", "content": "I need an assessment"}]},
        "expect_empty_recs": True,
    },
    {
        "label": "Java developer mid-level",
        "body": {"messages": [{"role": "user", "content": "Hiring a mid-level Java developer with 4 years experience"}]},
        "expect_recs": True,
    },
    {
        "label": "CXO leadership selection",
        "body": {"messages": [
            {"role": "user", "content": "We need assessments for CXO-level executives"},
            {"role": "assistant", "content": '{"reply":"Is this for selection or development?","recommendations":[],"end_of_conversation":false}'},
            {"role": "user", "content": "Selection against leadership benchmark"},
        ]},
        "expect_recs": True,
    },
    {
        "label": "Contact centre agents",
        "body": {"messages": [{"role": "user", "content": "Hiring contact centre agents, need spoken English assessment"}]},
        "expect_recs": True,
    },
    {
        "label": "Graduate financial analysts",
        "body": {"messages": [{"role": "user", "content": "Graduate financial analysts — numerical reasoning and personality"}]},
        "expect_recs": True,
    },
    {
        "label": "Off-topic refusal (HIPAA)",
        "body": {"messages": [{"role": "user", "content": "Is it HIPAA compliant to use SHL personality tests?"}]},
        "expect_empty_recs": True,
    },
    {
        "label": "Comparison: OPQ32r vs GSA",
        "body": {"messages": [{"role": "user", "content": "What is the difference between OPQ32r and the Global Skills Assessment?"}]},
    },
    {
        "label": "Refinement: add personality test",
        "body": {"messages": [
            {"role": "user", "content": "Hiring admin assistants who use Excel and Word"},
            {"role": "assistant", "content": '{"reply":"Here are knowledge tests.","recommendations":[{"name":"Microsoft Excel (New)","url":"https://www.shl.com/products/product-catalog/view/microsoft-excel-new/","test_type":"K"}],"end_of_conversation":false}'},
            {"role": "user", "content": "Also add a personality assessment to the list"},
        ]},
        "expect_recs": True,
    },
]


@dataclass
class TestResult:
    label: str
    status: int = 0
    elapsed_ms: float = 0.0
    recs: int = 0
    eoc: bool = False
    schema_ok: bool = False
    grounded: bool = False
    error: str = ""
    reply_preview: str = ""


async def send_one(client: httpx.AsyncClient, payload: dict) -> TestResult:
    result = TestResult(label=payload["label"])
    start = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}/chat", json=payload["body"], timeout=35.0)
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        result.status = resp.status_code

        if resp.status_code != 200:
            result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        data = resp.json()
        reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        eoc = data.get("end_of_conversation", None)

        result.schema_ok = (
            isinstance(reply, str) and bool(reply)
            and isinstance(recs, list)
            and isinstance(eoc, bool)
            and len(recs) <= 10
            and all(
                isinstance(r.get("name"), str)
                and isinstance(r.get("url"), str)
                and isinstance(r.get("test_type"), str)
                for r in recs
            )
        )

        result.grounded = all(
            r.get("url", "").startswith("https://www.shl.com/products/product-catalog/view/")
            for r in recs
        ) if recs else True

        result.recs = len(recs)
        result.eoc = bool(eoc)
        result.reply_preview = reply[:80].replace("\n", " ")

    except httpx.TimeoutException:
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        result.error = "TIMEOUT (>35 s)"
    except Exception as exc:
        result.elapsed_ms = (time.perf_counter() - start) * 1000
        result.error = str(exc)[:100]
    return result


async def run_load_test():
    print("\n" + "="*70)
    print("  SHL Recommender -- Load Test (v2)")
    print(f"  Target: {BASE_URL}")
    print(f"  Requests: {len(PAYLOADS)} | Pattern: 3-second stagger")
    print("="*70 + "\n")

    async with httpx.AsyncClient() as client:
        try:
            h = await client.get(f"{BASE_URL}/health", timeout=10)
            print(f"[health] {'OK' if h.status_code == 200 else 'FAIL'} -- {h.json()}")
            if h.status_code != 200:
                return False
        except Exception as e:
            print(f"[health] ERROR: {e} -- is the server running?")
            return False

    print(f"\nFiring {len(PAYLOADS)} requests with 3s stagger...\n")

    results: list[TestResult] = []
    overall_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        for i, payload in enumerate(PAYLOADS):
            if i > 0:
                await asyncio.sleep(3)
            print(f"  [{i+1}/{len(PAYLOADS)}] {payload['label']}...")
            result = await send_one(client, payload)
            results.append(result)

    overall_elapsed = (time.perf_counter() - overall_start) * 1000

    print("\n")
    print(f"{'#':<3} {'Label':<40} {'ms':>6} {'Recs':>5} {'Schema':>7} {'Ground':>7} {'EOC':>5} {'Result'}")
    print("-" * 82)

    passed = 0
    failed = 0

    for i, r in enumerate(results):
        ok = r.schema_ok and r.grounded and not r.error
        if ok:
            passed += 1
        else:
            failed += 1

        print(
            f"{i+1:<3} {r.label[:40]:<40} {r.elapsed_ms:>6.0f} {r.recs:>5} "
            f"{'YES' if r.schema_ok else 'NO':>7} "
            f"{'YES' if r.grounded else 'NO':>7} "
            f"{'T' if r.eoc else 'F':>5}  {'[OK]' if ok else '[FAIL]'}"
        )
        if r.error:
            print(f"    ERROR: {r.error}")
        elif r.reply_preview:
            print(f"    Reply: {r.reply_preview}...")

    print("\n" + "="*70)
    print(f"  Passed:  {passed}/{len(results)}")
    print(f"  Failed:  {failed}/{len(results)}")
    print(f"  Total wall clock:  {overall_elapsed/1000:.1f} s")
    avg = sum(r.elapsed_ms for r in results) / len(results)
    max_ms = max(r.elapsed_ms for r in results)
    print(f"  Avg latency: {avg:.0f} ms | Max: {max_ms:.0f} ms  {'[>30s WARN]' if max_ms > 30000 else '[<30s OK]'}")
    print("="*70 + "\n")
    return failed == 0


if __name__ == "__main__":
    ok = asyncio.run(run_load_test())
    sys.exit(0 if ok else 1)
